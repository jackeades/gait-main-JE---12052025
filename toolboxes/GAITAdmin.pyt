# -*- coding: utf-8 -*-
import os
import sys
from typing import Tuple, List, Optional

# https://devtopia.esri.com/ArcGISPro/Python/wiki/Custom-Python-Registry-Keys-and-Environment-Variables
sys.path.insert(
    0,
    os.path.join(
        os.environ["APPDATA"],
        "python",
        f"python{sys.version_info[0]}{sys.version_info[1]}",
        "site-packages"))

from collections import Counter

import arcpy
import pandas as pd
# The following imports are required for the toolbox to work
import transformers
import sentence_transformers
import wordninja
import torch

from gait import (
    FEL,
    FELMemory,
    FELFlow,
    FELObserverABC,
    FELLine,
    FEL0,
    FEL1,
    FEL2,
    Layers,
    Layer,
    Column,
)


class Toolbox:
    def __init__(self):
        self.label = "Toolbox"
        self.alias = "toolbox"
        self.tools = [
            FELDuckDB,
            FELPrepare,
            FELUpdate,
            FELSelect,
        ]


class FELDuckDB:
    def __init__(self):
        self.label = "Features to DuckDB"
        self.description = "Convert feature classes to DuckDB"
        self.exclude_types = (
            "oid",
            "guid",
            "globalid",
            "geometry",
        )
        self.exclude_names = (
            "globalid",
            "shape_length",
            "shape_area",
            "shape__length",
            "shape__area",
            "st_area(shape)",
            "st_perimeter(shape)",
        )

    def getParameterInfo(self):
        file_name = arcpy.Parameter(
            name="file_name",
            displayName="DuckDB Path",
            direction="Input",
            datatype="GPString",
            parameterType="Required",
            category="Advanced",
        )
        # file_name.value = os.path.expanduser(os.path.join("~", "layers.ddb"))
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        file_name.value = aprx.filePath.replace(".aprx", ".ddb")

        include_layer = arcpy.Parameter(
            displayName="Include Layer(s)",
            name="include_layer",
            datatype="GPFeatureLayer",
            parameterType="Optional",  # Parameter is optional
            direction="Input",
            multiValue=True  # Allows selecting multiple layers
        )

        exclude_layer = arcpy.Parameter(
            displayName="Exclude Layer(s)",
            name="exclude_layer",
            datatype="GPFeatureLayer",
            parameterType="Optional",  # Parameter is optional
            direction="Input",
            multiValue=True  # Allows selecting multiple layers
        )

        return [
            include_layer,
            exclude_layer,
            file_name,
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def _create_sp_ref_table(self, conn, sp_ref) -> None:
        wkid = sp_ref.factoryCode
        text = sp_ref.exportToString()
        conn.execute("""
        CREATE OR REPLACE TABLE sp_ref (
            wkid INTEGER,
            text TEXT
        )
        """)

        # Insert the Web Mercator spatial reference into the table
        conn.execute("""
                     INSERT INTO sp_ref (wkid, text)
                     VALUES (?, ?)
                     """, (wkid, text))

    def execute(self, parameters, messages):
        import duckdb

        arcpy.env.autoCancelling = False

        include_text = parameters[0].valueAsText
        exclude_text = parameters[1].valueAsText
        duckdb_path = parameters[2].valueAsText
        sp_ref = arcpy.env.outputCoordinateSystem
        if not sp_ref:
            sp_ref = arcpy.SpatialReference(4326)

        include_layers = include_text.split(";") if include_text else []
        exclude_layers = exclude_text.split(";") if exclude_text else []
        include_layers = [_.replace("'", "") for _ in include_layers]
        exclude_layers = [_.replace("'", "") for _ in exclude_layers]

        if os.path.exists(duckdb_path):
            arcpy.AddError(f"{duckdb_path} already exists.")
            return
        with duckdb.connect(duckdb_path) as conn:
            conn.execute("INSTALL spatial;LOAD spatial;")
            # Create a table to store the spatial reference
            self._create_sp_ref_table(conn, sp_ref)
            # Create a table for each feature class
            curr_proj = arcpy.mp.ArcGISProject("CURRENT")
            layer_list = curr_proj.activeMap.listLayers()
            for layer in layer_list:
                if layer.isGroupLayer or layer.isBroken or layer.isBasemapLayer:
                    continue
                if exclude_layers and layer.longName in exclude_layers:
                    continue
                if include_layers and layer.longName not in include_layers:
                    continue
                if not layer.isFeatureLayer:
                    continue
                if arcpy.env.isCancelled:
                    break

                arcpy.SetProgressorLabel(f"Processing {layer.name}...")
                table_name = layer.name.replace(" ", "_").replace("\\", "_")
                desc = arcpy.Describe(layer)
                shape_field = desc.shapeFieldName
                name_type = [
                    (f.name, f.type) for f in desc.fields
                    if f.type != "Geometry" and f.name.lower() not in self.exclude_names
                ]
                fields = [
                    name for (name, _) in name_type
                ]
                fields.append("SHAPE@WKB")  # Force WKB output
                columns = fields[:-1] + [shape_field]
                arcgis_to_pandas = {
                    "SmallInteger": "Int16",
                    "Integer": "Int32",
                    "Single": "float32",
                    "Double": "float64",
                    "String": "string",
                    "Date": "datetime64[ns]",
                    "OID": "Int32",
                    "GUID": "string",
                    "GlobalID": "string",
                    "Blob": "object",
                    "Raster": "object",
                    "XML": "object"
                }
                # Convert ArcGIS field types to pandas dtypes
                dtypes = {
                    name: arcgis_to_pandas.get(dtype, "object")  # Default to "object" if type is unknown
                    for name, dtype in name_type
                }
                # TODO - Use batch generator with progress bar if the number of records is too large.
                with arcpy.da.SearchCursor(layer, fields, spatial_reference=sp_ref) as cursor:
                    pdf = pd.DataFrame(cursor, columns=columns).astype(dtypes, errors="ignore")

                if arcpy.env.isCancelled:
                    break

                _ = conn.execute(f"""
                create or replace table {table_name} as
                SELECT * EXCLUDE({shape_field}),ST_GeomFromWKB({shape_field}) AS geom
                FROM pdf;
                
                CREATE INDEX {table_name}_index ON {table_name} USING RTREE (geom); 
                """)
                arcpy.AddMessage(f"Created Table {table_name}.")


class FELPrepare:
    def __init__(self):
        self.label = "Prepare Existing Locations"
        self.description = "Prepare Existing Locations"
        self.layers = []
        self.exclude_types = (
            "oid",
            "guid",
            "globalid",
            "geometry",
        )
        self.exclude_names = (
            "globalid",
            "shape_length",
            "shape_area",
            "shape__length",
            "shape__area",
            "st_area(shape)",
            "st_perimeter(shape)",
            "createdby",
            "createddate",
            "createdate",
            "modifiedby",
            "modifieddate",
            "lon",
            "lat",
            "longitude",
            "latitude",
            "point_x",
            "point_y",
        )
        self.max_method = "COUNT"
        self.max_records = 10000
        self.max_values = 50

    def getParameterInfo(self):
        include_layer = arcpy.Parameter(
            displayName="Include Layer(s)",
            name="include_layer",
            datatype=["GPFeatureLayer", "GPMapServerLayer", "GPFeatureRecordSetLayer", "GPInternetTiledLayer"],
            parameterType="Optional",  # Parameter is optional
            direction="Input",
            multiValue=True  # Allows selecting multiple layers
        )

        exclude_layer = arcpy.Parameter(
            displayName="Exclude Layer(s)",
            name="exclude_layer",
            datatype=["GPFeatureLayer", "GPMapServerLayer", "GPFeatureRecordSetLayer", "GPInternetTiledLayer"],
            parameterType="Optional",  # Parameter is optional
            direction="Input",
            multiValue=True  # Allows selecting multiple layers
        )

        meta_path = arcpy.Parameter(
            category="Advanced",
            name="meta_path",
            displayName="Metadata Path",
            direction="Input",
            datatype="GPString",
            parameterType="Required")
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        meta_path.value = aprx.filePath.replace(".aprx", ".json")

        max_values = arcpy.Parameter(
            category="Advanced",
            name="max_values",
            displayName="Max Values",
            direction="Input",
            datatype="GPLong",
            parameterType="Required")
        max_values.value = 50

        max_records = arcpy.Parameter(
            category="Advanced",
            name="max_records",
            displayName="Max Records",
            direction="Input",
            datatype="GPLong",
            parameterType="Required")
        max_records.value = 10000

        # Required parameter for selection type (TOP, LIMIT, COUNT)
        max_method = arcpy.Parameter(
            category="Advanced",
            displayName="Max Records Method",
            name="max_method",
            datatype="GPString",
            parameterType="Required",  # Required parameter
            direction="Input"
        )

        # Set the allowed values and default
        max_method.filter.type = "ValueList"
        max_method.filter.list = ["TOP", "LIMIT", "COUNT"]
        max_method.value = "COUNT"  # Default value

        max_samples = arcpy.Parameter(
            category="Advanced",
            name="max_samples",
            displayName="Max Samples",
            direction="Input",
            datatype="GPLong",
            parameterType="Required")
        max_samples.value = 100

        return [
            include_layer,
            exclude_layer,
            meta_path,
            max_method,
            max_records,
            max_values,
            max_samples,
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def _desc_columns(
            self,
            desc,
            # exclude_fields: List[str],
    ) -> list[Column]:
        columns = []
        for _ in desc.fields:
            # arcpy.AddMessage(f"{_.name} {_.type} {_.aliasName}")
            if _.type.lower() in self.exclude_types:
                continue
            if _.name.lower() in self.exclude_names:
                continue
            # if _.name.lower() in exclude_fields:
            #     continue

            if "_" in _.aliasName:
                alias = " ".join(wordninja.split(_.aliasName))
            else:
                alias = _.aliasName
            columns.append(Column(name=_.name,
                                  alias=alias.lower(),
                                  dtype=_.type
                                  ))
        return columns

    def _desc_values(self, layer, columns) -> None:
        match self.max_method:
            case "TOP":
                sql_clause = (f"TOP {self.max_records}", "")
            case "LIMIT":
                sql_clause = ("", f"LIMIT {self.max_records}")
            case _:
                sql_clause = ("", "")
        counters = [Counter() for _ in columns]
        try:
            with arcpy.da.SearchCursor(layer,
                                       [_.name for _ in columns],
                                       sql_clause=sql_clause,
                                       ) as cursor:
                for n, r in enumerate(cursor):
                    if n <= self.max_records:
                        for c, v in zip(counters, r):
                            if v is not None:
                                c[str(v).strip()] += 1
        except RuntimeError as e:  # TODO: Do we need this?
            arcpy.AddWarning(e)

        for column, counter in zip(columns, counters):
            column.values = [v for (v, _) in counter.most_common(self.max_values)]

    def _desc_layer(self, layer) -> None:
        desc = arcpy.Describe(layer)
        alias = desc.aliasName if hasattr(desc, "aliasName") else layer.name
        columns = self._desc_columns(desc)
        self._desc_values(layer, columns)
        self.layers.append(Layer(name=layer.longName,  # .replace(" ", "_").replace("\\", "_"),
                                 uri=layer.dataSource,  # if layer.isWebLayer else '',
                                 alias=alias.lower(),
                                 stype=desc.shapeType,
                                 columns=columns))

    def _desc_layer_list(
            self,
            layer_list,
            exclude_layers: list[str],
            include_layers: list[str],
    ):
        for layer in layer_list:
            arcpy.SetProgressorLabel(f"Processing {layer.name}...")
            if arcpy.env.isCancelled:
                break
            if layer.isGroupLayer or layer.isBroken or layer.isBasemapLayer or layer.isRasterLayer:
                continue
            if exclude_layers and layer.longName in exclude_layers:
                continue
            if include_layers and layer.longName not in include_layers:
                continue
            if layer.isFeatureLayer:
                self._desc_layer(layer)
            # elif layer.isWebLayer and hasattr(layer, "dataSource"):
            #     url = layer.dataSource
            #     cim_lyr = layer.getDefinition('V3')
            #     self.desc_layer_list(cim_lyr.subLayers, exclude_layers, include_layers)
            #     for sub_layer in cim_lyr.subLayers:
            #         if type(sub_layer).__name__ == 'CIMServiceCompositeSubLayer':
            #             for sub_sub_layer in sub_layer.subLayers:
            #                 self._desc_layer(layer)
            #         else:
            #             self._desc_layer(layer)

    def execute(self, parameters, messages):
        include_text = parameters[0].valueAsText
        exclude_text = parameters[1].valueAsText
        layers_json = parameters[2].valueAsText
        self.max_method = parameters[3].valueAsText
        self.max_records = parameters[4].value
        self.max_values = parameters[5].value
        max_samples = parameters[6].value

        arcpy.env.autoCancelling = False

        include_layers = include_text.split(";") if include_text else []
        exclude_layers = exclude_text.split(";") if exclude_text else []
        include_layers = [_.replace("'", "") for _ in include_layers]
        exclude_layers = [_.replace("'", "") for _ in exclude_layers]

        curr_proj = arcpy.mp.ArcGISProject("CURRENT")
        layer_list = curr_proj.activeMap.listLayers()

        self.layers = []
        self._desc_layer_list(layer_list, exclude_layers, include_layers)

        if not arcpy.env.isCancelled:
            device = "cuda" if arcpy.env.processorType == "GPU" and torch.cuda.is_available() else "cpu"
            arcpy.SetProgressorLabel(f"Creating Layer Files Using {device.upper()}...")

            l = Layers(layers=self.layers).prune()
            l.dump(layers_json)

            fel = FEL(layers=l)
            vss = FELMemory(
                device=device,
                overwrite_json_files=True,
            )
            fel1 = [fel.create_line_1().to_fel0() for _ in range(max_samples // 2)]
            fel2 = [fel.create_line_2().to_fel0() for _ in range(max_samples // 2)]
            vss.create_fel0(fel1 + fel2)
            vss.create_fel1([fel.create_line_1() for _ in range(max_samples)])
            vss.create_fel2([fel.create_line_2() for _ in range(max_samples)])
            vss.dump(layers_json)
            arcpy.AddMessage(f"Saved {layers_json}.")

    def postExecute(self, parameters):
        return


class FELUpdate:
    def __init__(self):
        self.label = "Update Existing Locations"
        self.description = "Update Existing Locations"

    def getParameterInfo(self):
        meta_path = arcpy.Parameter(
            name="meta_path",
            displayName="Metadata Path",
            direction="Input",
            datatype="File",
            parameterType="Required")
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        meta_path.value = aprx.filePath.replace(".aprx", ".json")

        return [
            meta_path,
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        meta_path = parameters[0].valueAsText
        device = "cuda" if arcpy.env.processorType == "GPU" and torch.cuda.is_available() else "cpu"
        vss = FELMemory(device=device)
        vss.load(meta_path)
        vss.dump(meta_path)

    def postExecute(self, parameters):
        return


class FELAddMessage(FELObserverABC):
    def on_fel_line(self, fel_line: FELLine) -> None:
        # arcpy.AddMessage(f"FEL Line: {fel_line}")
        return

    def on_fel0(self, fel0: FEL0) -> None:
        arcpy.AddMessage(f"FEL0: {fel0}")

    def on_fel1(self, fel1: FEL1) -> None:
        arcpy.AddMessage(f"FEL1: {fel1}")

    def on_fel2(self, fel2: FEL2) -> None:
        arcpy.AddMessage(f"FEL2: {fel2}")


class FELSelect:
    def __init__(self):
        self.label = "Select Existing Locations"
        self.description = "Select Existing Locations"

    def getParameterInfo(self):
        prompt = arcpy.Parameter(
            name="prompt",
            displayName="Prompt",
            direction="Input",
            datatype="String",
            parameterType="Required"
        )
        prompt.value = "Show all wells with depth greater than 200 meters"

        fel_path = arcpy.Parameter(
            name="fel_path",
            displayName="Metadata Path",
            direction="Input",
            datatype="File",
            parameterType="Required",
            category="Advanced",
        )
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        fel_path.value = aprx.filePath.replace(".aprx", ".json")

        model = arcpy.Parameter(
            name="model",
            displayName="Model",
            direction="Input",
            datatype="String",
            parameterType="Required",
            category="Advanced",
        )
        model.value = "azure/" + os.environ.get("AZURE_API_DEPLOYMENT", "gpt-4.1")

        api_base = arcpy.Parameter(
            name="api_base",
            displayName="API Base",
            direction="Input",
            datatype="String",
            parameterType="Required",
            category="Advanced",
        )
        api_base.value = os.environ.get("AZURE_API_BASE", "")

        api_version = arcpy.Parameter(
            name="api_version",
            displayName="API Version",
            direction="Input",
            datatype="String",
            parameterType="Required",
            category="Advanced",
        )
        api_version.value = os.environ.get("AZURE_API_VERSION", "")

        return [
            prompt,
            fel_path,
            model,
            api_base,
            api_version,
        ]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def _get_relation(
            self,
            relation: str
    ) -> Tuple[str, str]:
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

    def _get_linear_unit(
            self,
            distance: float,
            units: str
    ) -> str:
        return f"{distance} {units.capitalize()}" if distance else ""

    def _get_layer_by_name(
            self,
            layers: List[Layer],
            name: str
    ) -> Optional[Layer]:
        return next(filter(lambda _: _.name.lower() == name.lower(), layers), None)

    def execute(self, parameters, messages):
        prompt = parameters[0].valueAsText
        fel_path = parameters[1].valueAsText
        model = parameters[2].valueAsText
        api_base = parameters[3].valueAsText
        api_version = parameters[4].valueAsText
        device = "cuda" if arcpy.env.processorType == "GPU" and torch.cuda.is_available() else "cpu"
        try:
            fel_flow = FELFlow(
                fel_path,
                model,
                api_base=api_base,
                api_version=api_version,
                device=device,
                observer=FELAddMessage(),
            )
            fel = fel_flow(prompt)
            arcpy.AddMessage(str(fel))
            if isinstance(fel, FEL1):
                arcpy.management.SelectLayerByAttribute(fel.layer, "NEW_SELECTION", fel.where)
            elif isinstance(fel, FEL2):
                if fel.where1:
                    arcpy.management.SelectLayerByAttribute(fel.layer1, "NEW_SELECTION", fel.where1)

                if fel.where2:
                    arcpy.management.SelectLayerByAttribute(fel.layer2, "NEW_SELECTION", fel.where2)

                if fel.relation:
                    invert, overlap_type = self._get_relation(fel.relation)
                    search_distance = self._get_linear_unit(fel.distance, fel.unit)
                    arcpy.management.SelectLayerByLocation(fel.layer1,
                                                           overlap_type,
                                                           fel.layer2,
                                                           search_distance,
                                                           "SUBSET_SELECTION",
                                                           invert)

                    if fel.where2:
                        arcpy.management.SelectLayerByAttribute(fel.layer2, selection_type="CLEAR_SELECTION")
            else:
                arcpy.AddError(f"Invalid FEL Model.")
        except Exception as e:
            arcpy.AddError(f"Error: {e}")

    def postExecute(self, parameters):
        return
