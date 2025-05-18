"""
Script documentation

- Tool that uses Idris-generated SQL/NL pairs to query geospatial data
- Processes natural language queries and converts them to SQL
- Executes SQL queries against feature layers in ArcGIS Pro
"""
import os
import sys
import json
import glob
from typing import Optional, Tuple, List

import arcpy
import numpy as np
import re

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
import litellm

from gait import Idris, IdrisRDB, IdrisEmb, IdrisLLM
from gait.idris import IdrisLiteLLM, IdrisLiteEmb

# Spatial relation helpers copied from the FEL implementation
def _get_relation(relation: str) -> Tuple[str, str]:
    match relation:
        case "notIntersects":
            return "INVERT", "INTERSECT"
        case "withinDistance":
            return "NOT_INVERT", "WITHIN_A_DISTANCE"
        case "notWithinDistance":
            return "INVERT", "WITHIN_A_DISTANCE"
        case "contains":
            return "NOT_INVERT", "CONTAINS"
        case "notContains":
            return "INVERT", "CONTAINS"
        case "within":
            return "NOT_INVERT", "WITHIN"
        case "notWithin":
            return "INVERT", "WITHIN"
        case "near":
            return "NOT_INVERT", "WITHIN_A_DISTANCE_GEODESIC"
        case _:
            return "NOT_INVERT", "INTERSECT"


def _get_linear_unit(distance: float, units: str) -> str:
    """Return ArcPy linear unit string"""
    return f"{distance} {units.capitalize()}" if distance else ""

# Adds the custom module to the path if needed
custom_path = os.path.join(os.path.dirname(__file__), "..", "gait", "idris")
if custom_path not in sys.path:
    sys.path.append(custom_path)

# Then import normally in the appropriate place
from gait.idris.idris_precomputed import IdrisPrecomputedEmb


class ArcpyRDB(IdrisRDB):
    """RDB class that uses ArcPy to execute SQL against feature layers"""
    
    def __init__(self, layer_obj):
        self.layer = layer_obj
        self._dialect = "ESRI ArcObjects SQL"
    
    @property
    def dialect(self) -> str:
        return self._dialect
    
    def _get_create_table_columns(self, table_name: str) -> List[str]:
        # Return schema from layer
        desc = arcpy.Describe(self.layer.longName)
        return [f"{f.name} {f.type}" for f in desc.fields 
                if f.type not in ("OID", "Geometry", "GlobalID", "GUID")]
    
    def execute_sql(self, sql: str) -> None:
        """Execute SQL using ArcPy's SelectLayerByAttribute"""
        
        # Parse WHERE clause from SQL statement
        where_clause = sql
        
        # Check if this is a SELECT statement
        if sql.strip().upper().startswith("SELECT"):
            # Extract WHERE clause
            where_parts = sql.split("WHERE", 1)
            if len(where_parts) > 1:
                where_clause = where_parts[1].strip()
                # Remove any trailing parts like ORDER BY, GROUP BY, etc.
                for keyword in ["ORDER BY", "GROUP BY", "HAVING", "LIMIT"]:
                    if keyword in where_clause.upper():
                        where_clause = where_clause.split(keyword, 1)[0].strip()
            else:
                # No WHERE clause, select all features
                where_clause = "1=1"
        
        # Clean up the WHERE clause
        where_clause = where_clause.strip().rstrip(';')
        
        # Get available fields
        field_names = [field.name for field in arcpy.ListFields(self.layer.longName)]
        arcpy.AddMessage(f"Available fields: {field_names}")
        
        # Check if this is a date query
        date_query = False
        date_patterns = [
            r"(\w+)\s*=\s*'(\d{4})-(\d{2})-(\d{2})'",  # FIELD = '2025-05-01'
            r"CAST\s*\((\w+)[^)]*\)\s*=\s*'(\d{4})-(\d{2})-(\d{2})'"  # CAST(FIELD...) = '2025-05-01'
        ]
        
        # Check for date field in the layer
        date_fields = [field.name for field in arcpy.ListFields(self.layer.longName) 
                      if field.type == 'Date']
        
        # Only process date patterns if we have date fields
        if date_fields:
            arcpy.AddMessage(f"Date fields found: {date_fields}")
            for pattern in date_patterns:
                for match in re.finditer(pattern, where_clause, re.IGNORECASE):
                    if len(match.groups()) >= 4:
                        field_name = match.group(1)
                        year = match.group(2)
                        month = match.group(3)
                        day = match.group(4)
                        
                        # Check if the field exists (case-insensitive)
                        field_exists = False
                        for actual_field in field_names:
                            if actual_field.lower() == field_name.lower():
                                field_name = actual_field  # Use the correct case
                                field_exists = True
                                break
                        
                        if field_exists and field_name in date_fields:
                            date_query = True
                            # Replace with direct date comparison method
                            old_condition = match.group(0)
                            new_condition = f"{field_name} >= timestamp '{month}/{day}/{year} 00:00:00' AND {field_name} <= timestamp '{month}/{day}/{year} 23:59:59'"
                            where_clause = where_clause.replace(old_condition, new_condition)
        
        arcpy.AddMessage(f"Selecting features with WHERE clause: {where_clause}")
        
        try:
            # Select features using the WHERE clause
            arcpy.management.SelectLayerByAttribute(
                self.layer, 
                "NEW_SELECTION",
                where_clause
            )
            
            # Get the count of selected features
            result = arcpy.GetCount_management(self.layer)
            count = int(result.getOutput(0))
            arcpy.AddMessage(f"Selected {count} features")
            
        except Exception as e:
            arcpy.AddError(f"Error executing SQL: {str(e)}")
            
            # If it's a date query that failed, try an alternative approach
            if date_query and date_fields:
                try:
                    arcpy.AddMessage("Trying alternative date format")
                    # Extract month, day, year from original query if possible
                    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", where_clause)
                    if match:
                        year, month, day = match.groups()
                        for date_field in date_fields:
                            simple_where = f"{date_field} >= timestamp '{month}/{day}/{year} 00:00:00' AND {date_field} <= timestamp '{month}/{day}/{year} 23:59:59'"
                            try:
                                arcpy.AddMessage(f"Trying date field {date_field}: {simple_where}")
                                arcpy.management.SelectLayerByAttribute(self.layer, "NEW_SELECTION", simple_where)
                                result = arcpy.GetCount_management(self.layer)
                                count = int(result.getOutput(0))
                                arcpy.AddMessage(f"Selected {count} features with date field {date_field}")
                                return
                            except:
                                continue
                except Exception as date_e:
                    arcpy.AddError(f"Alternative date approach failed: {str(date_e)}")
            
            # Clear selection if all attempts failed
            arcpy.management.SelectLayerByAttribute(
                self.layer, 
                "CLEAR_SELECTION"
            )
    
    def close(self) -> None:
        pass


def load_idris(
    idris_folder: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = "2024-10-21"
) -> Optional[Idris]:
    """Load an Idris model from a folder created by PrepareIdrisGeospatialData.py"""
    
    if not os.path.exists(idris_folder):
        arcpy.AddError(f"Idris folder not found: {idris_folder}")
        return None
    
    # Check for required files
    create_table_file = os.path.join(idris_folder, "create_table.sql")
    context_file = os.path.join(idris_folder, "context.json")
    question_sql_file = os.path.join(idris_folder, "question_sql.json")
    
    if not all(os.path.exists(f) for f in [create_table_file, context_file, question_sql_file]):
        arcpy.AddError(f"Missing required files in {idris_folder}")
        return None
    
    # Updated environment variables to match PrepareIdrisGeospatialData.py
    os.environ["AZURE_OPENAI_API_KEY"] = api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
    os.environ["AZURE_OPENAI_API_VERSION"] = api_version
    
    # Original API environment variables
    os.environ["AZURE_API_KEY"] = api_key
    os.environ["AZURE_API_BASE"] = api_base
    os.environ["AZURE_API_VERSION"] = api_version
    os.environ["AZURE_API_DEPLOYMENT"] = model_name or "gpt-4o"

    os.environ["OPENAI_API_KEY"] = api_key

    api_endpoint = api_base
    if api_endpoint and not api_endpoint.startswith(('http://', 'https://')):
        api_endpoint = f"https://{api_endpoint}"
    
    emb_path = os.path.join(idris_folder, "embeddings")
    
    # Check if embeddings exist
    required_files = [
        "context_embeddings.npy", 
        "question_embeddings.npy",
        "context_data.json", 
        "question_sql_data.json"
    ]
    
    embedding_files_exist = (
        os.path.exists(emb_path) and 
        all(os.path.exists(os.path.join(emb_path, f)) for f in required_files)
    )
    
    if embedding_files_exist:
        try:
            # Direct manual loading to bypass potential issues in the class
            context_emb_file = os.path.join(emb_path, "context_embeddings.npy")
            question_emb_file = os.path.join(emb_path, "question_embeddings.npy")
            context_data_file = os.path.join(emb_path, "context_data.json") 
            question_data_file = os.path.join(emb_path, "question_sql_data.json")
            
            # Load the raw numpy arrays
            context_embeddings = np.load(context_emb_file, allow_pickle=True)
            question_embeddings = np.load(question_emb_file, allow_pickle=True)
            
            # Load JSON files with error handling for various encodings
            def load_json_safe(file_path):
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return json.load(f)
                    except UnicodeDecodeError:
                        continue
                # If all encodings fail, try binary mode
                with open(file_path, 'rb') as f:
                    return json.loads(f.read().decode('utf-8', errors='replace'))
            
            context_data = load_json_safe(context_data_file)
            question_sql_data = load_json_safe(question_data_file)
            
            arcpy.AddMessage(f"Loaded embeddings manually: context={context_embeddings.shape}, questions={question_embeddings.shape}")
            
            # Custom EmbbeddingModel that uses these preloaded values
            class ManualLoadedEmb(IdrisLiteEmb):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.context_embeddings = context_embeddings.tolist()
                    self.question_sql_embeddings = question_embeddings.tolist()
                    self.context = context_data
                    self.question_sql = question_sql_data
            
            # Use our custom class
            emb = ManualLoadedEmb(
                model_name="text-embedding-ada-002",
                api_base=api_endpoint,
                api_key=api_key,
                api_version=api_version,
                custom_llm_provider="azure"
            )
            
            arcpy.AddMessage(f"Embeddings loaded: context={len(emb.context_embeddings)}, questions={len(emb.question_sql_embeddings)}")
            
        except Exception as e:
            arcpy.AddError(f"Error loading embeddings directly: {str(e)}")
            arcpy.AddWarning("Falling back to on-the-fly embedding")
            # Fall back to default embedding method
            emb = IdrisLiteEmb(
                model_name="text-embedding-ada-002",
                api_base=api_endpoint,
                api_key=api_key,
                api_version=api_version,
                custom_llm_provider="azure"
            )
    else:
        arcpy.AddWarning("Precomputed embeddings not found - using on-the-fly embedding")
        # Fall back to default embedding method
        emb = IdrisLiteEmb(
            model_name="text-embedding-ada-002",
            api_base=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            custom_llm_provider="azure"
        )
    
    # Initialize LLM
    llm = IdrisLiteLLM(
        model_name=model_name or "gpt-4o",  # Use proper fallback
        api_base=api_endpoint,  # Use the corrected endpoint
        api_key=api_key,
        api_version=api_version,
        custom_llm_provider="azure"  # Added this parameter explicity
    )
    
    try:
        # Quick test to validate connections
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
    except Exception as e:
        arcpy.AddError(f"API connection test failed: {str(e)}")
        return None

    # Load data
    try:
        # Get create table statement
        with open(create_table_file, 'r', encoding='utf-8') as f:
            create_table = f.read()
        
        # Get context
        with open(context_file, 'r', encoding='utf-8') as f:
            context = json.load(f)
        
        # Get question/sql pairs
        with open(question_sql_file, 'r', encoding='utf-8') as f:
            question_sql = json.load(f)
        
        return create_table, context, question_sql, emb, llm
    
    except Exception as e:
        arcpy.AddError(f"Error loading Idris data: {str(e)}")
        return None


def find_layer_idris_folder(output_root: str, layer_name: str) -> Optional[str]:
    """Find the Idris folder for a given layer name"""
    
    # Try exact match first
    potential_folder = os.path.join(output_root, layer_name.replace(" ", "_"))
    if os.path.exists(potential_folder):
        return potential_folder
    
    # Try case-insensitive match
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path) and item.lower() == layer_name.lower().replace(" ", "_"):
            return item_path
    
    # Try partial match - if only one folder contains the layer name
    matching_folders = []
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path) and layer_name.lower().replace(" ", "_") in item.lower():
            matching_folders.append(item_path)
    
    if len(matching_folders) == 1:
        return matching_folders[0]
    
    return None


def run_query(
    query_text: str,
    layer_name: str,
    idris_root: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = "2024-10-21",
    layer2_name: Optional[str] = None,
    relation: str = "intersects",
    distance: float = 0.0,
    unit: str = "meters",
) -> bool:
    """
    Main function to run a natural language query on a geospatial layer using Idris.
    
    Args:
        query_text: Natural language query
        layer_name: Name of the layer to query
        idris_root: Root folder containing Idris data
        model_name: LLM model name
        api_key: API key for Azure/OpenAI
        api_base: API base URL for Azure/OpenAI
        api_version: API version for Azure/OpenAI
        layer2_name: Optional name of a second layer for spatial filtering
        relation: Spatial relationship to apply (e.g. "intersects", "withinDistance")
        distance: Buffer distance used with relation when applicable
        unit: Linear unit for the distance parameter
        
    Returns:
        bool: True if query executed successfully, False otherwise
    """
    arcpy.AddMessage(f"Processing query: {query_text}")
    arcpy.AddMessage(f"Target layer: {layer_name}")
    if layer2_name:
        arcpy.AddMessage(f"Spatial filter layer: {layer2_name}")
    
    # Get current project
    curr_proj = arcpy.mp.ArcGISProject("CURRENT")
    
    # Find the layer by name
    target_layer = None
    for layer in curr_proj.activeMap.listLayers():
        if layer.name.lower() == layer_name.lower():
            target_layer = layer
            break
    
    if not target_layer:
        arcpy.AddError(f"Layer '{layer_name}' not found in the active map")
        return False

    target_layer2 = None
    if layer2_name:
        for layer in curr_proj.activeMap.listLayers():
            if layer.name.lower() == layer2_name.lower():
                target_layer2 = layer
                break
        if not target_layer2:
            arcpy.AddError(f"Layer '{layer2_name}' not found in the active map")
            return False
    
    # Find the Idris data folder for this layer
    idris_folder = find_layer_idris_folder(idris_root, layer_name)
    
    if not idris_folder:
        arcpy.AddError(f"No Idris data found for layer '{layer_name}' in {idris_root}")
        return False
    
    arcpy.AddMessage(f"Using Idris data from: {idris_folder}")
    
    # Load Idris data
    idris_data = load_idris(
        idris_folder,
        model_name,
        api_key,
        api_base,
        api_version
    )
    
    if not idris_data:
        arcpy.AddError("Failed to load Idris data")
        return False
    
    create_table, context, question_sql, emb, llm = idris_data
    
    # Create RDB for ArcPy
    rdb = ArcpyRDB(target_layer)
    
    # Create Idris instance
    idris = Idris(rdb, emb, llm)
    
    # Initialize Idris with data
    idris.add_create_table(create_table)
    idris.load_context(context)
    #idris.load_question_sql(question_sql)
    
    # Process the query
    try:
        arcpy.AddMessage("Generating SQL from natural language query...")
        
        # Add debug info about the query process
        arcpy.AddMessage("Attempting to match query against available templates...")
        
        # Patch the generate_sql method to catch the specific error
        original_generate_sql = idris.generate_sql
        
        def debug_generate_sql(query):
            try:
                # Original code with error handling
                if not idris.idris_emb.question_sql:
                    arcpy.AddWarning("No question-SQL templates available for matching")
                    return None
                    
                arcpy.AddMessage(f"Using {len(idris.idris_emb.question_sql)} question templates")
                return original_generate_sql(query)
            except IndexError as e:
                # More generic fallback that works with any query
                arcpy.AddWarning(f"Index error during template matching: {str(e)}")
                
                # Extract key terms from the query
                terms = query.lower().split()
                search_terms = [term for term in terms if len(term) > 3]  # Skip small words
                
                # Try to find any similar questions
                for q, sql in idris.idris_emb.question_sql:
                    # Look for matching terms
                    matches = sum(1 for term in search_terms if term in q.lower())
                    if matches > 0:
                        arcpy.AddMessage(f"Found potential match: {q}")
                        return sql
                
                # If all else fails
                arcpy.AddWarning("No matches found, using generic query")
                return "SELECT * FROM " + rdb.layer.name
            except Exception as e:
                arcpy.AddWarning(f"Error during SQL generation: {str(e)}")
                return None
        
        # Replace the method temporarily
        idris.generate_sql = debug_generate_sql
        
        sql = idris.generate_sql(query_text)
        
        if sql:
            arcpy.AddMessage(f"Generated SQL: {sql}")
            
            # Execute the SQL (this will select features in the layer)
            rdb.execute_sql(sql)

            # Optionally refine the selection using a second layer
            if layer2_name:
                arcpy.management.MakeFeatureLayer(target_layer, "LHS")
                arcpy.management.MakeFeatureLayer(target_layer2, "RHS")
                invert, overlap_type = _get_relation(relation)
                search_distance = _get_linear_unit(distance, unit)
                arcpy.management.SelectLayerByLocation(
                    "LHS",
                    overlap_type,
                    "RHS",
                    search_distance,
                    "SUBSET_SELECTION",
                    invert,
                )
            return True
        else:
            arcpy.AddWarning("No SQL could be generated from the query")
            return False
    
    except Exception as e:
        arcpy.AddError(f"Error processing query: {str(e)}")
        return False
    
    finally:
        # Clean up resources
        idris.close()


if __name__ == "__main__":
    # Get tool parameters
    query_text = arcpy.GetParameterAsText(0)  # The Natural language query
    layer_name = arcpy.GetParameterAsText(1)  # The Layer to query
    idris_root = arcpy.GetParameterAsText(2)  # Root folder containing Idris data

    # LLM API parameters
    model_name = arcpy.GetParameterAsText(3)  # Model name
    api_key = arcpy.GetParameterAsText(4)  # API key
    api_base = arcpy.GetParameterAsText(5)  # API base URL
    api_version = arcpy.GetParameterAsText(6)  # API version

    # Optional spatial parameters
    layer2_name = arcpy.GetParameterAsText(7)
    relation = arcpy.GetParameterAsText(8)
    distance_param = arcpy.GetParameterAsText(9)
    unit = arcpy.GetParameterAsText(10)
    try:
        distance = float(distance_param) if distance_param else 0.0
    except ValueError:
        distance = 0.0
    
    # Run the query
    run_query(
        query_text,
        layer_name,
        idris_root,
        model_name,
        api_key,
        api_base,
        api_version,
        layer2_name,
        relation,
        distance,
        unit,
    )
