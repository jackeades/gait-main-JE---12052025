"""
Script documentation

- Tool parameters are accessed using arcpy.GetParameter() or 
                                     arcpy.GetParameterAsText()
- Update derived parameter values using arcpy.SetParameter() or
                                        arcpy.SetParameterAsText()
"""
import os
import sys
from collections import Counter
from typing import List

import arcpy

# https://devtopia.esri.com/ArcGISPro/Python/wiki/Custom-Python-Registry-Keys-and-Environment-Variables
sys.path.insert(
    0,
    os.path.join(
        os.environ["APPDATA"],
        "python",
        f"python{sys.version_info[0]}{sys.version_info[1]}",
        "site-packages"))

# The following imports are required for the toolbox to work

import wordninja
import torch
import sentence_transformers
import huggingface_hub

from gait import (
    Layer,
    Layers,
    Column,
    FEL,
    FELMemory,
)


def _desc_to_columns(
        desc,
        exclude_fields: List[str] = [],
) -> list[Column]:
    columns = []
    for _ in desc.fields:
        if _.type.lower() in (
                "oid",
                "guid",
                "globalid",
                "geometry",
        ):
            continue
        if _.name.lower() in (
                "globalid",
                "shape_length",
                "shape_area",
                "st_area(shape)",
                "st_perimeter(shape)",
                # hardcoded fields here - TODO: make this a parameter.
                "createdby",
                "createddate",
                "modifiedby",
                "modifieddate",
                "lon",
                "lat",
                "point_x",
                "point_y",
        ):
            continue
        if _.name.lower() in exclude_fields:
            continue

        if "_" in _.aliasName:
            alias = " ".join(wordninja.split(_.aliasName))
        else:
            alias = _.aliasName
        columns.append(Column(name=_.name,
                              alias=alias.lower(),
                              dtype=_.type
                              ))
    return columns


def _get_layer(layer_, layer_id: int) -> Layer:
    desc = arcpy.Describe(layer_.longName)
    alias = desc.aliasName if hasattr(desc, "aliasName") else layer_.name
    columns = _desc_to_columns(desc)
    return Layer(name=layer_.longName,
                 uri=layer_.dataSource,  # if layer.isWebLayer else '',
                 alias=alias.lower(),
                 stype=desc.shapeType,
                 columns=columns)


if __name__ == "__main__":
    arcpy.env.autoCancelling = False

    include_text = arcpy.GetParameterAsText(0)
    exclude_text = arcpy.GetParameterAsText(1)
    layers_json = arcpy.GetParameterAsText(2)
    max_method = arcpy.GetParameter(3)
    max_records = arcpy.GetParameter(4)
    max_values = arcpy.GetParameter(5)
    max_samples = arcpy.GetParameter(6)

    include_layers = include_text.split(";") if include_text else []
    exclude_layers = exclude_text.split(";") if exclude_text else []
    include_layers = [_.replace("'", "") for _ in include_layers]
    exclude_layers = [_.replace("'", "") for _ in exclude_layers]
    curr_proj = arcpy.mp.ArcGISProject("CURRENT")
    layer_list = curr_proj.activeMap.listLayers()
    layers = []
    for layer in layer_list:
        if layer.isGroupLayer or layer.isBroken or layer.isBasemapLayer or layer.isRasterLayer:
            continue
        if exclude_layers and layer.longName in exclude_layers:
            continue
        if include_layers and layer.longName not in include_layers:
            continue
        if layer.isFeatureLayer:
            layers.append(_get_layer(layer, len(layers)))
        elif layer.isWebLayer and hasattr(layer, "dataSource"):
            url = layer.dataSource
            cim_lyr = layer.getDefinition('V3')
            for sub_layer in cim_lyr.subLayers:
                if type(sub_layer).__name__ == 'CIMServiceCompositeSubLayer':
                    for sub_sub_layer in sub_layer.subLayers:
                        sub_layer_url = f"{url}/{sub_sub_layer.subLayerID}"
                        layers.append(_get_layer(layer, len(layers)))
                else:
                    sub_layer_url = f"{url}/{sub_layer.subLayerID}"
                    layers.append(_get_layer(layer, len(layers)))

    for layer in layers:
        arcpy.SetProgressorLabel(f"Processing {layer.alias}...")
        if arcpy.env.isCancelled:
            break
        fields = layer.columns_no_geom
        in_table = layer.uri or layer.name
        match max_method:
            case "TOP":
                sql_clause = (f"TOP {max_records}", "")
            case "LIMIT":
                sql_clause = ("", f"LIMIT {max_records}")
            case _:
                sql_clause = ("", "")
        counters = [Counter() for _ in fields]
        try:
            with arcpy.da.SearchCursor(in_table,
                                       [f.name for f in fields],
                                       sql_clause=sql_clause,
                                       ) as cursor:
                for n, r in enumerate(cursor):
                    if n <= max_records:
                        for c, v in zip(counters, r):
                            if v is not None:
                                c[str(v).strip()] += 1
        except RuntimeError as e:
            arcpy.AddWarning(e)

        for field, counter in zip(fields, counters):
            field.values = [v for (v, _) in counter.most_common(max_values)]

    if not arcpy.env.isCancelled:
        device = "cuda" if arcpy.env.processorType == "GPU" and torch.cuda.is_available() else "cpu"
        arcpy.SetProgressorLabel(f"Creating Layer Files Using {device.upper()}...")
        l = Layers(layers=layers).prune()
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
