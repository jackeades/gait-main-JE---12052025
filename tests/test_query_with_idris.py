import sys
import types
import importlib
import unittest
from unittest.mock import MagicMock, patch


def create_arcpy_mock():
    arcpy = types.SimpleNamespace()
    arcpy.AddMessage = MagicMock()
    arcpy.AddError = MagicMock()
    arcpy.AddWarning = MagicMock()

    mgmt = types.SimpleNamespace()
    mgmt.MakeFeatureLayer = MagicMock()
    mgmt.SelectLayerByLocation = MagicMock()
    mgmt.SelectLayerByAttribute = MagicMock()
    arcpy.management = mgmt

    arcpy.GetCount_management = MagicMock(return_value=types.SimpleNamespace(getOutput=lambda i: "1"))
    arcpy.ListFields = MagicMock(return_value=[])
    arcpy.Describe = MagicMock(return_value=types.SimpleNamespace(fields=[]))

    class DummyLayer:
        def __init__(self, name):
            self.name = name
            self.longName = name

    class DummyMap:
        def listLayers(self):
            return [DummyLayer("Layer1"), DummyLayer("Layer2")]

    class DummyProject:
        def __init__(self, path):
            self.activeMap = DummyMap()

    arcpy.mp = types.SimpleNamespace(ArcGISProject=lambda p: DummyProject(p))
    return arcpy


class DummyIdris:
    def __init__(self, rdb, emb, llm):
        self.rdb = rdb
        self.idris_emb = types.SimpleNamespace(question_sql=[("q", "SQL")])
    def add_create_table(self, table):
        pass
    def load_context(self, ctx):
        pass
    def generate_sql(self, query):
        return "SELECT * FROM table WHERE 1=1"
    def close(self):
        pass


class QueryWithIdrisTests(unittest.TestCase):
    def test_run_query_multiple_layers(self):
        arcpy_mock = create_arcpy_mock()
        dummy_gait = types.ModuleType("gait")
        dummy_gait.Idris = DummyIdris
        dummy_gait.IdrisRDB = object
        dummy_gait.IdrisEmb = object
        dummy_gait.IdrisLLM = object
        dummy_gait.idris = types.ModuleType("gait.idris")
        dummy_gait.idris.IdrisLiteLLM = object
        dummy_gait.idris.IdrisLiteEmb = object
        dummy_gait.idris.idris_precomputed = types.ModuleType("gait.idris.idris_precomputed")
        dummy_gait.idris.idris_precomputed.IdrisPrecomputedEmb = object

        patched_modules = {
            "arcpy": arcpy_mock,
            "numpy": MagicMock(),
            "litellm": MagicMock(),
            "wordninja": MagicMock(),
            "gait": dummy_gait,
            "gait.idris": dummy_gait.idris,
            "gait.idris.idris_precomputed": dummy_gait.idris.idris_precomputed,
        }
        with patch.dict(sys.modules, patched_modules):
            with patch.dict('os.environ', {"APPDATA": "/tmp"}):
                qwi = importlib.import_module("toolboxes.QueryWithIdris")
            with patch.object(qwi, "find_layer_idris_folder", return_value="/tmp"):
                with patch.object(qwi, "load_idris", return_value=("", "", "", None, None)):
                    with patch.object(qwi, "Idris", DummyIdris):
                        result = qwi.run_query(
                            query_text="find reports",
                            layer_name="Layer1",
                            idris_root="root",
                            model_name="model",
                        api_key="key",
                        api_base="base",
                        api_version="v",
                        layer2_name="Layer2",
                        relation="intersects",
                        distance=0.0,
                        unit="meters",
                    )
                    self.assertTrue(result)
                    self.assertTrue(qwi.arcpy.management.SelectLayerByLocation.called)
                    self.assertTrue(qwi.arcpy.management.SelectLayerByAttribute.called)


if __name__ == "__main__":
    unittest.main()
