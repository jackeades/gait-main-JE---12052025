import os
from typing import Tuple, Dict, Any

import arcpy
import requests


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


def _get_linear_unit(
        distance: float,
        units: str
) -> str:
    return f"{distance} {units.capitalize()}" if distance else ""


def _fel1(fel: Dict[str, Any]) -> None:
    arcpy.management.MakeFeatureLayer(fel["layer"], "LHS", fel["where"] or "1=1")
    out_fc = os.path.join(arcpy.env.scratchGDB, "GAIT")
    arcpy.management.CopyFeatures("LHS", out_fc)
    arcpy.SetParameter(0, out_fc)


def _fel2(fel: Dict[str, Any]) -> None:
    arcpy.management.MakeFeatureLayer(fel["layer1"], "LHS", fel["where1"] or "1=1")
    arcpy.management.MakeFeatureLayer(fel["layer2"], "RHS", fel["where2"] or "1=1")
    invert, overlap_type = _get_relation(fel["relation"])
    search_distance = _get_linear_unit(fel["distance"], fel["unit"])
    arcpy.management.SelectLayerByLocation("LHS",
                                           overlap_type,
                                           "RHS",
                                           search_distance,
                                           "SUBSET_SELECTION",
                                           invert)
    out_fc = os.path.join(arcpy.env.scratchGDB, "GAIT")
    arcpy.management.CopyFeatures("LHS", out_fc)
    arcpy.SetParameter(0, out_fc)


if __name__ == "__main__":
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False
    prompt = arcpy.GetParameterAsText(1)
    processing_server = arcpy.env.processingServer or "http://localhost:8000/fel"
    try:
        response = requests.get(
            processing_server,
            timeout=15,  # Timeout in seconds
            params={"prompt": prompt},
        )
        if response.status_code == 200:
            # Parse the JSON response
            resp_json = response.json()
            arcpy.AddMessage(f"JSON response: {resp_json}")
            if "layer" in resp_json:
                _fel1(resp_json)
            elif "layer1" in resp_json and "layer2" in resp_json:
                _fel2(resp_json)
            else:
                arcpy.AddError("JSON does not match FEL1 or FEL2 structure.")
        else:
            arcpy.AddError(f"Request failed with status code: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        arcpy.AddError(f"Failed to establish a connection: {e}")
    except requests.exceptions.Timeout:
        arcpy.AddError("The request timed out after 15 seconds.")
    except requests.exceptions.RequestException as e:
        arcpy.AddError(f"An error occurred: {e}")
    except Exception as e:
        arcpy.AddError(f"An error occurred: {e}")
