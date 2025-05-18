#
# fastapi run main.py --app app
#
# pkill -f fastapi
#
# https://jnikenoueba.medium.com/optimizing-performance-with-fastapi-c86206cb9e64
#
import os

import sys

# https://devtopia.esri.com/ArcGISPro/Python/wiki/Custom-Python-Registry-Keys-and-Environment-Variables
if "APPDATA" in os.environ:
    sys.path.insert(
        0,
        os.path.join(
            os.environ["APPDATA"],
            "python",
            f"python{sys.version_info[0]}{sys.version_info[1]}",
            "site-packages"))

# import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated
from typing import List, Tuple, Any

import geopandas as gpd
import pandas as pd
import uvicorn
import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from gait import Layer, FELFlow, FEL1, FEL2


@dataclass
class Settings:
    """General GAIT settings for applications.

    WARNING: Do not use this class directly. Use `get_settings()` instead.
    """
    # Override the default settings by setting the content of .env file.
    title: str
    version: str
    # log_level: int
    layers_json: str
    model_name: str
    api_base: str
    api_version: str


@lru_cache
def _get_settings(path: str = "./.env") -> Settings:
    """Get GAIT settings from the environment.

    :param path: The path to the .env file.
    :return: The GAIT settings.
    """
    from environs import Env

    env = Env()
    if os.path.exists(path):
        env.read_env(path)
    return Settings(
        title=env.str("APP_TITLE", "G.AI.T"),
        version=env.str("APP_VERSION", "0.1"),
        # log_level=env.log_level("LOG_LEVEL", logging.WARNING),
        layers_json=env.str("LAYERS_JSON", "layers.json"),
        model_name=env.str("MODEL_NAME", "azure/" + os.environ.get("AZURE_API_DEPLOYMENT", "gpt-4o")),
        api_base=env.str("API_BASE", os.environ.get("AZURE_API_BASE", "https://api.openai.com/v1/chat/completions")),
        api_version=env.str("API_VERSION", os.environ.get("AZURE_API_VERSION", "2024-10-21")),
    )


@lru_cache
def _get_flow() -> FELFlow:
    settings = _get_settings()
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return FELFlow(
        fel_path=settings.layers_json,
        model=settings.model_name,
        api_base=settings.api_base,
        api_version=settings.api_version,
        device=device,
    )


def _get_layers() -> List[Layer]:
    return _get_flow().layers


@asynccontextmanager
async def lifespan(app_: FastAPI):
    _get_settings()
    _get_flow()
    yield  # Start responding to FastAPI requests.


# logging.basicConfig(
#     level=_get_settings().log_level,
#     # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     format="%(message)s"
# )

app = FastAPI(docs_url=None,
              title=_get_settings().title,
              version=_get_settings().version,
              lifespan=lifespan
              )
app.mount("/webapp", StaticFiles(directory="webapp"), name="webapp")
# Set up CORSMiddleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def home() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get(
    "/health",
    description="Health Check.",
    summary="Health Check.",
    operation_id="get_health",
    tags=["System"],
)
async def get_health():
    return {"health": "ok"}


@app.get(
    "/settings",
    description="Get the configuration settings.",
    summary="Get settings.",
    operation_id="get_settings_dict",
    tags=["System"],
)
async def get_settings_dict() -> dict[str, Any]:
    return asdict(_get_settings())


@app.get("/docs", include_in_schema=False)
async def get_docs():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title,
        swagger_favicon_url="/favicon.ico"
    )


@lru_cache
def _get_favicon() -> FileResponse:
    file_name = "favicon.ico"
    file_path = os.path.join("static", file_name)
    return FileResponse(path=file_path, headers={"Content-Disposition": f"attachment; filename={file_name}"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return _get_favicon()


@lru_cache
def _get_fel(prompt: str) -> dict[str, Any]:
    fel = _get_flow()(prompt)
    match fel:
        case FEL1():
            fel.layer = _get_layer_by_name(fel.layer).uri
        case FEL2():
            fel.layer1 = _get_layer_by_name(fel.layer1).uri
            fel.layer2 = _get_layer_by_name(fel.layer2).uri
    return fel.model_dump()


@app.get("/fel",
         description="Find existing location.",
         summary="Find existing location.",
         operation_id="get_fel",
         tags=["GAIT"],
         )
async def get_fel(
        prompt: str
) -> dict[str, Any]:
    try:
        return _get_fel(prompt)
    except Exception as e:
        # logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


EXCLUDE = [
    "LON", "LAT",
    "POINT_X", "POINT_Y",
    "CREATEDBY", "CREATEDDATE",
    "EDITEDBY", "EDITEDDATE",
    "MODIFIEDBY", "MODIFIEDDATE",
    "FOLIO",
]


def _geo_interface(
        layer: str,
        where: str = "1=1",
) -> dict:
    """Convert an ArcGIS layer to a GeoJSON dictionary.

    :param layer: The full URI of the layer.
    :param where: The SQL where clause to filter the layer.
    """
    import arcpy

    desc = arcpy.Describe(layer)
    wkid = desc.spatialReference.factoryCode
    # TODO - Make EXCLUDE a settings parameter.
    fields = [_.name for _ in desc.fields if _.name not in EXCLUDE]
    tab = arcpy.da.TableToArrowTable(layer, fields, where, "WKB")
    pdf = tab.to_pandas()
    pdf.rename(columns={desc.shapeFieldName: 'geometry'}, inplace=True)
    pdf.geometry = pdf.geometry.apply(bytes)
    pdf.geometry = gpd.GeoSeries.from_wkb(pdf.geometry)
    gdf = gpd.GeoDataFrame(pdf, crs=f"EPSG:{wkid}")
    gdf = gdf.to_crs("EPSG:4326")
    return gdf.__geo_interface__


def _geo_json(
        layer: str,
        where: str = "1=1",
) -> dict:
    import arcpy

    desc = arcpy.Describe(layer)
    a_fields = [_.name for _ in desc.fields if _.type not in ['Geometry'] and _.name not in EXCLUDE]
    p_fields = a_fields.copy()
    p_fields.append("geometry")
    a_fields.append("SHAPE@WKB")
    sp_ref = arcpy.SpatialReference(4326)
    with arcpy.da.SearchCursor(desc.name,
                               a_fields,
                               where_clause=where,
                               spatial_reference=sp_ref
                               ) as sc:
        pdf = pd.DataFrame(sc, columns=p_fields)
    pdf.geometry = pdf.geometry.apply(bytes)
    pdf.geometry = gpd.GeoSeries.from_wkb(pdf.geometry)
    # Convert datetime columns to ISO format.
    columns = pdf.select_dtypes(include=['datetime64[ns]']).columns
    pdf[columns] = pdf[columns].applymap(lambda _: _.isoformat() if pd.notnull(_) else None)
    gdf = gpd.GeoDataFrame(pdf, crs="EPSG:4326")
    return {
        "type": "FeatureCollection",
        "features": [_ for _ in gdf.iterfeatures(drop_id=True)]
    }


def _get_fel1(fel: FEL1) -> dict:
    return _geo_interface(fel.layer, fel.where)


def _get_layer_by_name(name: str) -> Layer | None:
    return next(filter(lambda _: _.name.lower() == name.lower(), _get_layers()), None)


@app.get("/fel1",
         description="Find existing location 1.",
         summary="Find existing location 1.",
         operation_id="get_fel1",
         tags=["GAIT"],
         )
async def get_fel1(
        name: str,
        where: str = "1=1",
) -> dict:
    try:
        layer = _get_layer_by_name(name)
        if layer:
            fel = FEL1(layer=layer.uri, where=where)
            return _get_fel1(fel)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid layer: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def _get_fel2(fel: FEL2) -> dict:
    import arcpy

    arcpy.env.overwriteOutput = True
    arcpy.management.MakeFeatureLayer(fel.layer1, "LHS", fel.where1 or "1=1")
    arcpy.management.MakeFeatureLayer(fel.layer2, "RHS", fel.where2 or "1=1")
    invert, overlap_type = _get_relation(fel.relation)
    search_distance = _get_linear_unit(fel.distance, fel.unit)
    arcpy.management.SelectLayerByLocation("LHS",
                                           overlap_type,
                                           "RHS",
                                           search_distance,
                                           "NEW_SELECTION",
                                           invert)
    return _geo_interface("LHS")


@app.get("/fel2",
         description="Find existing location 2.",
         summary="Find existing location 2.",
         operation_id="get_fel2",
         tags=["GAIT"],
         )
async def get_fel2(
        name1: str,
        name2: str,
        where1: str = "1=1",
        where2: str = "1=1",
        relation: str = "intersects",
        distance: str = "0",
        unit: str = "meters"
) -> dict:
    try:
        layer1 = _get_layer_by_name(name1)
        layer2 = _get_layer_by_name(name2)
        if layer1 and layer2:
            fel = FEL2(
                layer1=layer1.uri,
                where1=where1,
                layer2=layer2.uri,
                where2=where2,
                relation=relation,
                distance=float(distance),
                unit=unit
            )
            return _get_fel2(fel)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid layer: {name1} or {name2}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute",
          description="Execute the user prompt.",
          summary="Execute prompt.",
          operation_id="post_execute",
          tags=["GAIT"],
          )
async def post_execute(
        question: Annotated[str, Form()]
) -> dict:
    try:
        fel = _get_flow()(question)
        match fel:
            case FEL1():
                name = fel.layer
                # Convert the name of the layer to the URI.
                layer = _get_layer_by_name(name)
                if layer:
                    fel.layer = layer.uri
                    data = _get_fel1(fel)
                    return dict(data=data, question=question, name=name, messsages=[])
                else:
                    raise HTTPException(status_code=400, detail=f"Cannot find layer: {fel.layer}")
            case FEL2():
                name = fel.layer1
                layer1 = _get_layer_by_name(fel.layer1)
                layer2 = _get_layer_by_name(fel.layer2)
                if layer1 and layer2:
                    fel.layer1 = layer1.uri
                    fel.layer2 = layer2.uri
                    data = _get_fel2(fel)
                    return dict(data=data, question=question, name=name, messsages=[])
                else:
                    raise HTTPException(status_code=400, detail=f"Cannot find layer: {fel.layer1} or {fel.layer2}")
            case _:
                raise HTTPException(status_code=400, detail=f"Invalid FEL type: {fel}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Google: create self signed certificate
    # TODO: Update instructions on fastapi to use the certificate.
    uvicorn.run(app,
                # TODO: Get host, port, ssl_xxx from settings.
                host="0.0.0.0",
                port=8000,
                ssl_certfile=os.path.expanduser(os.path.join("~", "cert.crt")),
                ssl_keyfile=os.path.expanduser(os.path.join("~", "cert.key"))
                )
