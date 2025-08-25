"""
@author: Dr.Lt.Data
@title: Impact Pack
@nickname: Impact Pack
@description: This extension offers various detector nodes and detailer nodes that allow you to configure a workflow that automatically enhances facial details. And provide iterative upscaler.
"""

import folder_paths
import os
import sys
import logging

comfy_path = os.path.dirname(folder_paths.__file__)
impact_path = os.path.join(os.path.dirname(__file__))
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)

import impact.config
# Only show loading message if verbose logging is enabled
if os.environ.get("COMFYUI_VERBOSE", "0") == "1":
    logging.info(f"### Loading: ComfyUI-Impact-Pack ({impact.config.version})")

# Core
# recheck dependencies for colab
try:
    import folder_paths
    import torch                  # noqa: F401
    import cv2                    # noqa: F401
    from cv2 import setNumThreads # noqa: F401
    import numpy as np            # noqa: F401
    import comfy.samplers
    import comfy.sd               # noqa: F401
    from PIL import Image, ImageFilter             # noqa: F401
    from skimage.measure import label, regionprops # noqa: F401
    from collections import namedtuple             # noqa: F401
    import piexif                                  # noqa: F401
    import nodes
except Exception as e:
    import logging
    logging.error("[Impact Pack] Failed to import due to several dependencies are missing!!!!")
    raise e


import impact.impact_server  # to load server api

# Only import the modules needed for the 7 specific nodes
from .modules.impact.impact_pack import *       # noqa: F403 - for SAMLoader and DetailerForEachPipe
from .modules.impact.detectors import *         # noqa: F403 - for SimpleDetectorForEach
from .modules.impact.pipe import *              # noqa: F403 - for ToBasicPipe
from .modules.impact.segs_nodes import *        # noqa: F403 - for SEGSToImageList, SEGSOrderedFilter, ControlNetApplySEGS

import threading


NODE_CLASS_MAPPINGS = {
    "SAMLoader": SAMLoader, # noqa: F405
    "ImpactSimpleDetectorSEGS": SimpleDetectorForEach, # noqa: F405
    "DetailerForEachPipe": DetailerForEachPipe, # noqa: F405
    "ToBasicPipe": ToBasicPipe, # noqa: F405
    "ImpactControlNetApplyAdvancedSEGS": ControlNetApplyAdvancedSEGS, # noqa: F405
    "SEGSToImageList": SEGSToImageList, # noqa: F405
    "ImpactSEGSOrderedFilter": SEGSOrderedFilter, # noqa: F405
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMLoader": "SAMLoader (Impact)",
    "ImpactSimpleDetectorSEGS": "Simple Detector (SEGS)",
    "DetailerForEachPipe": "Detailer (SEGS/pipe)",
    "ToBasicPipe": "ToBasicPipe",
    "ImpactControlNetApplyAdvancedSEGS": "ControlNetApply (Advanced SEGS)",
    "SEGSToImageList": "SEGS to Image List",
    "ImpactSEGSOrderedFilter": "SEGS Filter (ordered)",
}


# NOTE:  Inject directly into EXTENSION_WEB_DIRS instead of WEB_DIRECTORY
#        Provide the js path fixed as ComfyUI-Impact-Pack instead of the path name, making it available for external use

# WEB_DIRECTORY = "js"  -- deprecated method
# Only register the web directory if it exists to avoid server errors
_impact_js_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'js')
if os.path.isdir(_impact_js_dir):
    nodes.EXTENSION_WEB_DIRS["ComfyUI-Impact-Pack"] = _impact_js_dir


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
