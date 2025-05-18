"""
Script documentation

- Tool parameters are similar to PrepareExistingLocations.py
- Leverages the Idris module to create rich SQL/NL pairs for geospatial data
- Outputs can be used with tools similar to SelectExistingLocation.py
"""
import os
import sys
import json
import sqlite3
import openai
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

import arcpy
import pandas as pd
import numpy as np
import random

# ---- Remap user-supplied env vars to what the OpenAI/Azure SDK expects ----
# ---- Also Normalise whatever the user supplies (Azure or OpenAI) ---------------
def _configure_openai_env():
    """
    Normalises creds for BOTH providers.
    """
    if os.getenv("OPENAI_API_KEY"):          # plain OpenAI
        return

    az_key     = os.getenv("AZURE_API_KEY")
    az_base    = os.getenv("AZURE_API_BASE")      # must start with https:// to work with Azure
    az_version = os.getenv("AZURE_API_VERSION")

    if not (az_key and az_base and az_version):
        raise RuntimeError("Missing OpenAI or Azure credentials")

    # ★★ the line we forgot ★★
    os.environ["AZURE_OPENAI_ENDPOINT"]    = az_base     
    os.environ["AZURE_OPENAI_API_KEY"]     = az_key
    os.environ["AZURE_OPENAI_API_VERSION"] = az_version

    # generic alias (helps older libs, not used for routing)
    os.environ["OPENAI_API_KEY"] = az_key


is_azure = os.getenv("AZURE_OPENAI_ENDPOINT") is not None
azure_endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT")   # full https://.../gpt-4o etc etc
azure_version   = os.getenv("AZURE_OPENAI_API_VERSION")

# https://devtopia.esri.com/ArcGISPro/Python/wiki/Custom-Python-Registry-Keys-and-Environment-Variables
sys.path.insert(
    0,
    os.path.join(
        os.environ["APPDATA"],
        "python",
        f"python{sys.version_info[0]}{sys.version_info[1]}",
        "site-packages"))

# Required imports for Idris functionality
import wordninja
import torch
import litellm

from gait import Idris, IdrisTrainer, IdrisRDB, IdrisEmb, IdrisLLM
from gait.idris import IdrisLiteLLM, IdrisLiteEmb
from langchain_community.utilities import SQLDatabase

# Add the system prompt here
PARAPHRASE_SYS_PROMPT = """
You are an analyst writing questions to a GIS assistant.
For each input query, return {k} paraphrases that feel like they were
typed by *different* humans: sometimes terse, sometimes wordy; include
occasional abbreviations ("avg", "pop.") or minor typos, but NEVER
change the meaning.

IMPORTANT: For queries involving time or dates, preserve the exact time references:
- Time terms like "after 4pm", "before noon", "morning", "evening" must be kept intact
- Date references like "yesterday", "last week", "May 1st" must remain as is

Respond with a numbered list only.
"""

class MockRDB(IdrisRDB):
    """Mock RDB class for Idris to use during training"""
    
    @property
    def dialect(self) -> str:
        return "SQLite"
    
    def _get_create_table_columns(self, table_name: str) -> List[str]:
        # Will be populated during processing
        return []
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        arcpy.AddMessage("Would execute: " + sql)
        # Return an empty DataFrame with expected columns
        return pd.DataFrame()
    
    def close(self) -> None:
        pass


def _desc_to_dataframe(layer_obj, limit: int = 1000) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """Convert an ArcGIS layer to a pandas DataFrame with appropriate metadata"""
    
    arcpy.AddMessage("Processing " + layer_obj.name + "...")
    
    # Get layer description
    desc = arcpy.Describe(layer_obj.longName)
    
    # Prepare field info
    field_names = []
    field_aliases = {}
    
    # Filter out geometry and system fields
    exclude_types = ["OID", "Geometry", "GlobalID", "GUID"]
    exclude_names = [
        "OBJECTID", "SHAPE", "SHAPE_Length", "SHAPE_Area",
        "GlobalID", "CreatedBy", "CreatedDate", "ModifiedBy", "ModifiedDate"
    ]
    
    for field in desc.fields:
        if field.type in exclude_types or field.name in exclude_names:
            continue
        
        field_names.append(field.name)
        
        # Create human-readable aliases using wordninja
        if "_" in field.aliasName:
            alias = " ".join(wordninja.split(field.aliasName))
        else:
            alias = field.aliasName
        
        field_aliases[field.name] = alias.lower()
    
    # If no valid fields, return empty frame
    if not field_names:
        return pd.DataFrame(), desc.name, {}
    
    # Read data from feature class
    data_rows = []
    count = 0
    with arcpy.da.SearchCursor(layer_obj.longName, field_names) as cursor:
        for row in cursor:
            data_rows.append(row)
            count += 1
            if limit is not None and count >= limit:
                break
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=field_names)
    
    return df, desc.name, field_aliases


def create_idris_data(
    layer_obj,
    output_folder: str,
    max_records: int = 1000,
    max_pairs: int = 20,
    augmentation_count: int = 3,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = "2024-10-21"
) -> str:
    """Process a layer into Idris format with SQL/NL pairs"""
    
    # Create output folder if it doesn't exist
    layer_folder = os.path.join(output_folder, layer_obj.name.replace(" ", "_"))
    os.makedirs(layer_folder, exist_ok=True)
    
    # Convert layer to DataFrame with metadata
    df, table_name, aliases = _desc_to_dataframe(layer_obj, max_records)
    
    if df.empty:
        arcpy.AddWarning(f"No valid fields found in {layer_obj.name}")
        return None
    
    os.environ["AZURE_OPENAI_API_KEY"] = api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
    os.environ["AZURE_OPENAI_API_VERSION"] = api_version
    
    # These are still needed for backwards compatibility with the function
    os.environ["AZURE_API_KEY"] = api_key
    os.environ["AZURE_API_BASE"] = api_base
    os.environ["AZURE_API_VERSION"] = api_version
    os.environ["AZURE_API_DEPLOYMENT"] = model_name or "gpt-4o"

    # Set the OpenAI API key as fallback
    os.environ["OPENAI_API_KEY"] = api_key

    #remap to azure esri settings
    #_configure_openai_env()  

    arcpy.AddMessage("SDK key visible?" + str(bool(openai.api_key or os.getenv("OPENAI_API_KEY"))))
    
    # Create a SQLite database for LangChain augmentation
    db_path = os.path.join(layer_folder, f"{table_name}.db")
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    langchain_db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # Initialize IdrisTrainer with aliases
    trainer = IdrisTrainer(alias={f"_col:{k}": v for k, v in aliases.items()})
    
    # Generate initial SQL/NL pairs
    arcpy.AddMessage("Generating initial SQL/NL pairs for " + layer_obj.name + "...")
    result = trainer.train(df, table_name, limit=max_pairs if max_pairs is not None else 20)
    
    arcpy.AddMessage("Initial pairs count: " + str(len(result.question_sql)))
    
    # Add date query examples if date fields are detected
    date_fields = [field.name for field in arcpy.ListFields(layer_obj.longName) 
                  if field.type in ('Date')]
    
    if date_fields:
        arcpy.AddMessage(f"Adding time query examples for fields: {date_fields}")
        for field_name in date_fields:
            # Get sample dates from the actual data
            sample_dates = []
            try:
                with arcpy.da.SearchCursor(layer_obj.longName, [field_name]) as cursor:
                    for i, row in enumerate(cursor):
                        if row[0] and len(sample_dates) < 3:
                            # Convert to string in case it's a datetime object
                            date_str = str(row[0])
                            # Extract just the date part
                            date_part = date_str.split()[0] if ' ' in date_str else date_str
                            if date_part not in sample_dates:
                                sample_dates.append(date_part)
                        if i > 100:  # Don't scan too many rows
                            break
            except Exception as e:
                arcpy.AddWarning(f"Couldn't extract sample dates: {str(e)}")
            
            # Use a default date if we couldn't get samples
            if not sample_dates:
                sample_dates = ["2025-05-01"]
            
            # Use the first sample date for examples
            reference_date = sample_dates[0]
            
            # Add sample time queries using actual data format to ensure arcgis format is correct
            result.question_sql.extend([
                (f"Show {table_name} after 2pm", f"SELECT * FROM {table_name} WHERE {field_name} > timestamp '{reference_date} 14:00:00' AND {field_name} < timestamp '{reference_date} 23:59:59'"),
                (f"Find {table_name} before noon", f"SELECT * FROM {table_name} WHERE {field_name} > timestamp '{reference_date} 00:00:00' AND {field_name} < timestamp '{reference_date} 12:00:00'"),
                (f"Show {table_name} in the afternoon", f"SELECT * FROM {table_name} WHERE {field_name} > timestamp '{reference_date} 12:00:00' AND {field_name} < timestamp '{reference_date} 18:00:00'"),
            ])
            
            # Add context with a note about the reference date to ensure arcgis format is correct
            result.context.extend([
                f"For time-based queries on {field_name}, use timestamp comparisons:",
                f"1. 'After 2pm': {field_name} > timestamp 'YYYY-MM-DD 14:00:00'",
                f"2. The date part (YYYY-MM-DD) can be any valid date, as the query focuses on the time component",
                f"3. Time ranges should use consistent date parts: {field_name} > timestamp 'YYYY-MM-DD HH:MM:SS' AND {field_name} < timestamp 'YYYY-MM-DD HH:MM:SS'"
            ])
    
    # Augment pairs with LangChain if specified
    all_pairs = result.question_sql  # Default to just using original pairs

    if augmentation_count and augmentation_count > 0:
        arcpy.AddMessage("Augmenting pairs with LangChain...")
        try:
            # Convert possible None values to safe defaults
            max_pairs_to_augment = 10
            safe_model_name = model_name or "gpt-4o"
            safe_augmentation_count = augmentation_count or 3
            
            # Debugging information
            arcpy.AddMessage("Using model: " + safe_model_name)
            arcpy.AddMessage("Augmentation count: " + str(safe_augmentation_count))
            arcpy.AddMessage("Max pairs to augment: " + str(max_pairs_to_augment))
            
            # Get a safe subset of pairs to augment
            pairs_to_augment = random.sample(
                result.question_sql,
                k=min(len(result.question_sql), max_pairs or len(result.question_sql))
            )
            arcpy.AddMessage("Will augment " + str(len(pairs_to_augment)) + " pairs")
            
            # Call the augmentation method with all values guaranteed non-None
            aug_pairs = trainer.augment_pairs_with_langchain(
                pairs_to_augment,
                langchain_db,
                model_name=safe_model_name,
                paraphrases_per_query=safe_augmentation_count,
                system_prompt=PARAPHRASE_SYS_PROMPT
            )
            
            arcpy.AddMessage("Generated " + str(len(aug_pairs)) + " augmented pairs")
            
            # Combine original and augmented pairs
            all_pairs = result.question_sql + aug_pairs
        except Exception as e:
            arcpy.AddWarning(f"Error during augmentation: {str(e)}")
            # Continue using just the original pairs (no augmentation)
    
    # Save outputs
    arcpy.AddMessage("Saving results to " + layer_folder + "...")

    # Save CREATE TABLE statement
    with open(os.path.join(layer_folder, "create_table.sql"), "w") as f:
        f.write(result.create_table)
    
    # Save context
    with open(os.path.join(layer_folder, "context.json"), "w") as f:
        json.dump(result.context, f, indent=2)
    
    # Save question/sql pairs
    with open(os.path.join(layer_folder, "question_sql.json"), "w") as f:
        json.dump(all_pairs, f, indent=2)
    
    # Save human-readable pairs for inspection
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "layer_name": layer_obj.name,
        "table_name": table_name,
        "pairs": [
            {
                "natural_language": nl,
                "sql": sql
            }
            for nl, sql in all_pairs
        ]
    }
    
    with open(os.path.join(layer_folder, f"sql_nl_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(output_data, f, indent=2)


    try:
        # Generate Idris model data
        rdb = MockRDB()
        
        # Debug the API parameters
        arcpy.AddMessage("=== Parameter Debug Information ===")
        arcpy.AddMessage(f"Direct api_key: {bool(api_key)}")
        arcpy.AddMessage(f"Direct api_base: '{api_base}'")
        arcpy.AddMessage(f"Direct api_version: '{api_version}'")
        arcpy.AddMessage(f"Direct model_name: '{model_name}'")
        
        # Debug environment variables
        arcpy.AddMessage("=== Environment Variables Debug ===")
        arcpy.AddMessage(f"AZURE_OPENAI_API_KEY set: {bool(os.environ.get('AZURE_OPENAI_API_KEY'))}")
        arcpy.AddMessage(f"AZURE_OPENAI_ENDPOINT: '{os.environ.get('AZURE_OPENAI_ENDPOINT')}'")
        arcpy.AddMessage(f"AZURE_OPENAI_API_VERSION: '{os.environ.get('AZURE_OPENAI_API_VERSION')}'")
        
        # Ensure API base has https:// prefix
        api_endpoint = api_base
        if api_endpoint and not api_endpoint.startswith(('http://', 'https://')):
            api_endpoint = f"https://{api_endpoint}"
        

        # Remove the test values - use the actual parameters
        emb = IdrisLiteEmb(
            model_name="text-embedding-ada-002", 
            #model_name="text-embedding-3-small",
            api_base=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            custom_llm_provider="azure" 
        )

        llm = IdrisLiteLLM(
            model_name=model_name or "gpt-4o",
            api_base=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            custom_llm_provider="azure"
        )

        try:
            # Quick test to validate 
            arcpy.AddMessage("Testing API connection...")
            chat_resp = litellm.completion(
                model="azure/gpt-4o",
                api_base=api_endpoint,
                api_key=api_key,
                api_version=api_version,
                messages=[{"role": "user", "content": "ping"}],
                custom_llm_provider="azure"
            )
            
            arcpy.AddMessage("Chat test OK → " + chat_resp.choices[0].message.content)

            # Quick embed test
            emb_resp = litellm.embedding(
                model="azure/text-embedding-ada-002",
                api_base=api_endpoint,
                api_key=api_key,
                api_version=api_version,
                input="ping",
                custom_llm_provider="azure"
            )
            
            # Print the full structure to debug
            arcpy.AddMessage(f"Embed response structure: {type(emb_resp)}")
            arcpy.AddMessage(f"Embed response data: {type(emb_resp.data[0])}")
            
            # Access the embedding correctly based on the structure
            embedding_data = emb_resp.data[0]
            if hasattr(embedding_data, 'embedding'):
                arcpy.AddMessage(f"Embed dims = {len(embedding_data.embedding)}")
            else:
                arcpy.AddMessage(f"Embed dims = {len(embedding_data['embedding'])}")

        except Exception as e:
            arcpy.AddError("API connection test failed:\n" + str(e))
            raise
        
 
        
        # Create and initialize Idris instance
        idris = Idris(rdb, emb, llm)
        idris.add_create_table(result.create_table)
        idris.load_context(result.context)
        idris.load_question_sql(all_pairs)
        
        # Save the embeddings in the chosen folder
        emb_path = os.path.join(layer_folder, "embeddings")
        os.makedirs(emb_path, exist_ok=True)

        try:
            # Save the context and question/SQL data
            context_data = idris.idris_emb.dump_context()
            question_sql_data = idris.idris_emb.dump_question_sql()
            
            # Add debug lines - delete later:
            arcpy.AddMessage(f"DEBUG: About to save {len(question_sql_data)} question/SQL pairs")
            arcpy.AddMessage(f"DEBUG: About to save {len(idris.idris_emb.question_sql_embeddings)} question embeddings")
            
            with open(os.path.join(emb_path, "context_data.json"), "w") as f:
                json.dump(context_data, f)
            
            with open(os.path.join(emb_path, "question_sql_data.json"), "w") as f:
                json.dump(question_sql_data, f)
            
            # Save the embeddings directly from the instance
            # Convert to numpy arrays first for efficient storage
            context_embeddings = np.array(idris.idris_emb.context_embeddings)
            question_embeddings = np.array(idris.idris_emb.question_sql_embeddings)
            
            np.save(os.path.join(emb_path, "context_embeddings.npy"), context_embeddings)
            np.save(os.path.join(emb_path, "question_embeddings.npy"), question_embeddings)
            
            arcpy.AddMessage(f"Successfully saved {len(context_embeddings)} context embeddings and {len(question_embeddings)} question embeddings")
        except Exception as e:
            arcpy.AddWarning(f"Error saving embeddings: {str(e)}")
            import traceback
            arcpy.AddWarning(traceback.format_exc())

        # Return the path regardless of API connection success
        return layer_folder
    except Exception as e:
        arcpy.AddWarning(f"Error in Idris model generation: {str(e)}")
        # Still return the layer folder even if there was an error
        
    # Return the path regardless of API connection success
    return layer_folder


if __name__ == "__main__":
    arcpy.env.autoCancelling = False
    
    # Get tool parameters
    include_text = arcpy.GetParameterAsText(0)  # Layers to include
    exclude_text = arcpy.GetParameterAsText(1)  # Layers to exclude
    output_folder = arcpy.GetParameterAsText(2)  # Output folder
    max_records = arcpy.GetParameter(3)  # Max records per layer
    max_pairs = arcpy.GetParameter(4)  # Max SQL/NL pairs per field
    augmentation_count = arcpy.GetParameter(5)  # Number of augmented pairs per original
    
    # LLM API parameters
    model_name = arcpy.GetParameterAsText(6)  # Model name
    api_key = arcpy.GetParameterAsText(7)  # API key
    api_base = arcpy.GetParameterAsText(8)  # API base URL
    api_version = arcpy.GetParameterAsText(9)  # API version
    
    # Debug parameters at entry point
    arcpy.AddMessage("=== Main Function Parameters ===")
    arcpy.AddMessage(f"API Key provided: {bool(api_key)}")
    arcpy.AddMessage(f"API Base URL: '{api_base}'")
    arcpy.AddMessage(f"API Version: '{api_version}'")
    
    # Parse include/exclude lists
    include_layers = include_text.split(";") if include_text else []
    exclude_layers = exclude_text.split(";") if exclude_text else []
    include_layers = [_.replace("'", "") for _ in include_layers]
    exclude_layers = [_.replace("'", "") for _ in exclude_layers]
    
    # Get current project and map
    curr_proj = arcpy.mp.ArcGISProject("CURRENT")
    layer_list = curr_proj.activeMap.listLayers()
    
    # Process each layer
    processed_layers = []
    for layer in layer_list:
        if arcpy.env.isCancelled:
            break
            
        # Skip layers that don't meet criteria
        if (layer.isGroupLayer or layer.isBroken or 
            layer.isBasemapLayer or layer.isRasterLayer):
            continue
            
        if exclude_layers and layer.longName in exclude_layers:
            continue
            
        if include_layers and layer.longName not in include_layers:
            continue
            
        # Process feature layers
        if layer.isFeatureLayer:
            arcpy.SetProgressorLabel(f"Processing {layer.name}...")
            
            try:
                layer_folder = create_idris_data(
                    layer,
                    output_folder,
                    max_records,
                    max_pairs,
                    augmentation_count,
                    model_name,
                    api_key,
                    api_base,
                    api_version
                )
                
                if layer_folder:
                    processed_layers.append((layer.name, layer_folder))
                    arcpy.AddMessage("Successfully processed " + layer.name + " to " + layer_folder)
                
            except Exception as e:
                arcpy.AddError(f"Error processing {layer.name}: {str(e)}")
    
    # Create a summary file
    if processed_layers:
        summary_file = os.path.join(output_folder, f"idris_layers_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_layers": [
                    {
                        "name": name,
                        "folder": folder
                    } for name, folder in processed_layers
                ]
            }, f, indent=2)
        
        arcpy.AddMessage(f"Summary saved to {summary_file}")
        arcpy.AddMessage(f"Successfully processed {len(processed_layers)} layers")
    else:
        arcpy.AddMessage("No layers were processed") 