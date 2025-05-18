"""
Script documentation

- Tool parameters are accessed using arcpy.GetParameter() or 
                                     arcpy.GetParameterAsText()
- Update derived parameter values using arcpy.SetParameter() or
                                        arcpy.SetParameterAsText()
"""
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.environ["APPDATA"],
        "python",
        f"python{sys.version_info[0]}{sys.version_info[1]}",
        "site-packages"))

import arcpy
import torch
import sentence_transformers
import huggingface_hub

from gait import FELMemory

if __name__ == "__main__":
    fel_path = arcpy.GetParameterAsText(0)
    device = "cuda" if arcpy.env.processorType == "GPU" and torch.cuda.is_available() else "cpu"
    vss = FELMemory(device=device)
    vss.load(fel_path)
    vss.dump(fel_path)
