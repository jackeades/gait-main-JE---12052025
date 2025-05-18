from typing import Tuple

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


if __name__ == "__main__":
    prompt = arcpy.GetParameterAsText(0)
    processing_server = arcpy.env.processingServer or "http://localhost:8000/fel"
    try:
        response = requests.get(
            processing_server,
            timeout=15,  # Timeout in seconds
            params={"prompt": prompt},
        )
        if response.status_code == 200:
            # Parse the JSON response
            resp = response.json()
            arcpy.AddMessage(f"JSON response: {resp}")
            if "layer" in resp:
                arcpy.management.SelectLayerByAttribute(resp["layer"], "NEW_SELECTION", resp["where"] or "1=1")
            elif "layer1" in resp and "layer2" in resp:
                if resp["where1"]:
                    arcpy.management.SelectLayerByAttribute(resp["layer1"], "NEW_SELECTION", resp["where1"])

                if resp["where2"]:
                    arcpy.management.SelectLayerByAttribute(resp["layer2"], "NEW_SELECTION", resp["where2"])

                if resp["relation"]:
                    invert, overlap_type = _get_relation(resp["relation"])
                    search_distance = _get_linear_unit(resp["distance"], resp["unit"])
                    arcpy.management.SelectLayerByLocation(resp["layer1"],
                                                           overlap_type,
                                                           resp["layer2"],
                                                           search_distance,
                                                           "SUBSET_SELECTION",
                                                           invert)

                    if resp["where2"]:
                        arcpy.management.SelectLayerByAttribute(resp["layer2"], "CLEAR_SELECTION")
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
        arcpy.AddError(f"An unexpected error occurred: {e}")
