"""
Python toolbox for Idris-based Geospatial NL-to-SQL tools

This toolbox provides tools to:
1. Prepare Idris data from geospatial layers (PrepareIdrisGeospatialData)
2. Query geospatial layers using natural language (QueryWithIdris)
"""
import os
import arcpy


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "Idris Geospatial Tools"
        self.alias = "idristools"
        self.tools = [PrepareIdrisGeospatialData, QueryWithIdris]


class PrepareIdrisGeospatialData(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Prepare Idris Geospatial Data"
        self.description = "Prepares Idris SQL/NL pairs from geospatial data in ArcGIS Pro"
        self.canRunInBackground = True
        self.category = "Idris Tools"

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        
        # Input Layers (multi-value)
        param_include = arcpy.Parameter(
            displayName="Layers to Include",
            name="include_layers",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        param_include.filter.type = "ValueList"
        param_include.filter.list = [layer.name for layer in 
                                    arcpy.mp.ArcGISProject("CURRENT").activeMap.listLayers() 
                                    if not layer.isGroupLayer]
        params.append(param_include)
        
        # Excluded Layers (multi-value)
        param_exclude = arcpy.Parameter(
            displayName="Layers to Exclude",
            name="exclude_layers",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        param_exclude.filter.type = "ValueList"
        param_exclude.filter.list = [layer.name for layer in 
                                    arcpy.mp.ArcGISProject("CURRENT").activeMap.listLayers() 
                                    if not layer.isGroupLayer]
        params.append(param_exclude)
        
        # Output Folder
        param_output = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_output)
        
        # Maximum Records
        param_max_records = arcpy.Parameter(
            displayName="Max Records per Layer",
            name="max_records",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_max_records.value = 1000
        params.append(param_max_records)
        
        # Maximum SQL/NL pairs per field
        param_max_pairs = arcpy.Parameter(
            displayName="Max SQL/NL Pairs per Field",
            name="max_pairs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_max_pairs.value = 20
        params.append(param_max_pairs)
        
        # Augmentation Count
        param_augmentation = arcpy.Parameter(
            displayName="Augmentation Count per Original Pair",
            name="augmentation_count",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_augmentation.value = 3
        params.append(param_augmentation)
        
        # Model Name
        param_model = arcpy.Parameter(
            displayName="LLM Model Name",
            name="model_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_model.value = "gpt-4o"
        params.append(param_model)
        
        # API Key
        param_key = arcpy.Parameter(
            displayName="API Key",
            name="api_key",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_key)
        
        # API Base URL
        param_base = arcpy.Parameter(
            displayName="API Base URL",
            name="api_base",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_base)
        
        # API Version
        param_version = arcpy.Parameter(
            displayName="API Version",
            name="api_version",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_version.value = "2024-10-21"
        params.append(param_version)
        
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get parameters
        include_layers = parameters[0].valueAsText
        exclude_layers = parameters[1].valueAsText
        output_folder = parameters[2].valueAsText
        max_records = parameters[3].value
        max_pairs = parameters[4].value
        augmentation_count = parameters[5].value
        model_name = parameters[6].valueAsText
        api_key = parameters[7].valueAsText
        api_base = parameters[8].valueAsText
        api_version = parameters[9].valueAsText
        
        # Run the tool script
        script_path = os.path.join(os.path.dirname(__file__), "PrepareIdrisGeospatialData.py")
        
        # Import the module
        import importlib.util
        import datetime
        import json
        
        spec = importlib.util.spec_from_file_location("PrepareIdrisGeospatialData", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Ensure API base has https:// prefix if provided
        if api_base and not api_base.startswith(('http://', 'https://')):
            api_base = f"https://{api_base}"
            
        # Configure environment variables for OpenAI/Azure
        if api_key and api_base:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version
            
            # These are still needed for backwards compatibility
            os.environ["AZURE_API_KEY"] = api_key
            os.environ["AZURE_API_BASE"] = api_base
            os.environ["AZURE_API_VERSION"] = api_version
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Call the proper configuration function
            module._configure_openai_env()
            
        # Parse include/exclude lists
        include_list = include_layers.split(";") if include_layers else []
        exclude_list = exclude_layers.split(";") if exclude_layers else []
        include_list = [_.replace("'", "") for _ in include_list]
        exclude_list = [_.replace("'", "") for _ in exclude_list]
        
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
                
            if exclude_list and layer.longName in exclude_list:
                continue
                
            if include_list and layer.longName not in include_list:
                continue
                
            # Process feature layers
            if layer.isFeatureLayer:
                arcpy.SetProgressorLabel(f"Processing {layer.name}...")
                
                try:
                    # Directly call the create_idris_data function
                    layer_folder = module.create_idris_data(
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
            summary_file = os.path.join(output_folder, f"idris_layers_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.datetime.now().isoformat(),
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
        
        return


class QueryWithIdris(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Query with Idris"
        self.description = "Query geospatial data using natural language and Idris"
        self.canRunInBackground = False
        self.category = "Idris Tools"

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        
        # Natural Language Query
        param_query = arcpy.Parameter(
            displayName="Natural Language Query",
            name="query",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_query.value = "Show me all features where..."
        params.append(param_query)
        
        # Target Layer
        param_layer = arcpy.Parameter(
            displayName="Layer to Query",
            name="layer_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_layer.filter.type = "ValueList"
        param_layer.filter.list = [layer.name for layer in
                                  arcpy.mp.ArcGISProject("CURRENT").activeMap.listLayers()
                                  if not layer.isGroupLayer and layer.isFeatureLayer]
        params.append(param_layer)

        # Optional second layer for spatial filtering
        param_layer2 = arcpy.Parameter(
            displayName="Spatial Filter Layer",
            name="layer2_name",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_layer2.filter.type = "ValueList"
        param_layer2.filter.list = param_layer.filter.list
        params.append(param_layer2)

        # Spatial relation
        param_relation = arcpy.Parameter(
            displayName="Spatial Relation",
            name="relation",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_relation.filter.type = "ValueList"
        param_relation.filter.list = [
            "intersects",
            "notIntersects",
            "withinDistance",
            "notWithinDistance",
            "contains",
            "notContains",
            "within",
            "notWithin",
            "near",
        ]
        param_relation.value = "intersects"
        params.append(param_relation)

        # Distance for relation
        param_distance = arcpy.Parameter(
            displayName="Distance",
            name="distance",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        param_distance.value = 0
        params.append(param_distance)

        # Unit for distance
        param_unit = arcpy.Parameter(
            displayName="Distance Unit",
            name="unit",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        param_unit.filter.type = "ValueList"
        param_unit.filter.list = ["meters", "feet", "kilometers", "miles"]
        param_unit.value = "meters"
        params.append(param_unit)
        
        # Idris Data Root
        param_idris_root = arcpy.Parameter(
            displayName="Idris Data Root Folder",
            name="idris_root",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_idris_root)
        
        # Model Name
        param_model = arcpy.Parameter(
            displayName="LLM Model Name",
            name="model_name",
            datatype="GPString",
            parameterType="Required", 
            direction="Input"
        )
        param_model.value = "gpt-4o"
        params.append(param_model)
        
        # API Key
        param_key = arcpy.Parameter(
            displayName="API Key",
            name="api_key",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_key)
        
        # API Base URL
        param_base = arcpy.Parameter(
            displayName="API Base URL",
            name="api_base",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        params.append(param_base)
        
        # API Version
        param_version = arcpy.Parameter(
            displayName="API Version",
            name="api_version",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_version.value = "2024-10-21"
        params.append(param_version)
        
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get parameters
        query_text = parameters[0].valueAsText
        layer_name = parameters[1].valueAsText
        idris_root = parameters[2].valueAsText
        model_name = parameters[3].valueAsText
        api_key = parameters[4].valueAsText
        api_base = parameters[5].valueAsText
        api_version = parameters[6].valueAsText
        layer2_name = parameters[7].valueAsText if len(parameters) > 7 else None
        relation = parameters[8].valueAsText if len(parameters) > 8 else "intersects"
        distance = float(parameters[9].value) if len(parameters) > 9 and parameters[9].valueAsText else 0.0
        unit = parameters[10].valueAsText if len(parameters) > 10 else "meters"
        
        # Ensure API base has https:// prefix if provided
        if api_base and not api_base.startswith(('http://', 'https://')):
            api_base = f"https://{api_base}"
        
        # Configure environment variables for OpenAI/Azure
        if api_key and api_base:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version
            
            # These are still needed for backwards compatibility
            os.environ["AZURE_API_KEY"] = api_key
            os.environ["AZURE_API_BASE"] = api_base
            os.environ["AZURE_API_VERSION"] = api_version
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Import the QueryWithIdris module
        try:
            script_path = os.path.join(os.path.dirname(__file__), "QueryWithIdris.py")
            import importlib.util
            spec = importlib.util.spec_from_file_location("QueryWithIdris", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call the run_query function directly with the parameters
            success = module.run_query(
                query_text=query_text,
                layer_name=layer_name,
                idris_root=idris_root,
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                layer2_name=layer2_name,
                relation=relation,
                distance=distance,
                unit=unit,
            )
            
            if not success:
                arcpy.AddWarning("Query execution completed with warnings or errors.")
                
        except Exception as e:
            arcpy.AddError(f"Error executing tool: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            
        return 