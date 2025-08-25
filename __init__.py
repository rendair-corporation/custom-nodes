

# Use relative import - changed from 'from nodes' to 'from .nodes'
from .nodes import InputCollector, FloatToInt
from .image_filter import Image_Filters, Combine_Mask, Blur_Mask   
from .image_resize import ImageResize, ImageComposite
from .show_anything import showAnything
from .crop_inpainting import PrepareImageAndMaskForInpaint, OverlayInpaintedLatent, OverlayInpaintedImage
from .openai_node import OpenAINode
from .switch_any import CSwitchBooleanAny, CSwitchFromAny
from .image_contrast_adaptive_sharpening import ImageCAS
from .text_nodes import JoinWithDelimiter, CR_TextConcatenate
from aiohttp import web
from server import PromptServer
# LoraTagLoader is loaded via subdirectory mechanism below


# Import node mappings from subdirectories
import sys
import os
import traceback

def _is_verbose():
    return os.environ.get('COMFYUI_VERBOSE', '0') == '1' or os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true')

def safe_import_submodule(module_name, display_name):
    """Safely import node mappings from a submodule.

    Supports standard Python package names and folders containing hyphens by
    falling back to a file-based import of that folder's __init__.py.
    """
    try:
        import importlib
        module = importlib.import_module(f".{module_name}", package=__name__)

        class_mappings = getattr(module, 'NODE_CLASS_MAPPINGS', {})
        display_mappings = getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {})

        if _is_verbose():
            print(f"✓ Loaded {len(class_mappings)} nodes from {display_name}")
        return class_mappings, display_mappings

    except Exception as import_err:
        # Fallback for directories that are not valid module names (e.g., contain '-')
        try:
            import importlib.util
            base_dir = os.path.dirname(__file__)
            pkg_dir = os.path.join(base_dir, module_name)
            init_py = os.path.join(pkg_dir, "__init__.py")
            if not os.path.exists(init_py):
                raise FileNotFoundError(f"__init__.py not found in {pkg_dir}")

            sanitized = module_name.replace('-', '_')
            full_name = f"{__name__}.{sanitized}"
            spec = importlib.util.spec_from_file_location(full_name, init_py, submodule_search_locations=[pkg_dir])
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {display_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)

            class_mappings = getattr(module, 'NODE_CLASS_MAPPINGS', {})
            display_mappings = getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {})

            if _is_verbose():
                print(f"✓ Loaded {len(class_mappings)} nodes from {display_name}")
            return class_mappings, display_mappings

        except Exception as e:
            if _is_verbose():
                print(f"⚠ Failed to load {display_name}: {str(e)}")
            if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                traceback.print_exc()
            return {}, {}

# Import from subdirectories
subdirectory_mappings = []
subdirectories = [
    ("ComfyUI_IPAdapter_plus", "ComfyUI IPAdapter Plus"),
    ("comfyui-inpaint-nodes", "ComfyUI Inpaint Nodes"),
    ("ComfyUI-Detail-Daemon", "ComfyUI Detail Daemon"),
    ("ComfyUI-TiledDiffusion", "ComfyUI Tiled Diffusion"),
    ("comfyui_controlnet_aux", "ComfyUI ControlNet Auxiliary"),
    ("sd-perturbed-attention", "SD Perturbed Attention"),
    ("ComfyUI-Impact-Pack", "ComfyUI Impact Pack"),
    ("ComfyUI-Impact-Subpack", "ComfyUI Impact Subpack"),
    ("comfyui_lora_tag_loader", "ComfyUI LoRA Tag Loader"),
    ("ComfyUI_Anyline_main", "ComfyUI Anyline"),
]

for module_name, display_name in subdirectories:
    class_mappings, display_mappings = safe_import_submodule(module_name, display_name)
    if class_mappings:
        subdirectory_mappings.append((class_mappings, display_mappings))




# Base node mappings from main directory
NODE_CLASS_MAPPINGS = {
    "InputCollector": InputCollector,
    "Image_Filters": Image_Filters,
    "ImageResize": ImageResize,
    "ImageComposite": ImageComposite,
    "showAnything": showAnything,
    "PrepareImageAndMaskForInpaint": PrepareImageAndMaskForInpaint,
    "OverlayInpaintedLatent": OverlayInpaintedLatent,
    "OverlayInpaintedImage": OverlayInpaintedImage,
    "OpenAINode": OpenAINode,
    "CSwitchFromAny": CSwitchFromAny,
    "CSwitchBooleanAny": CSwitchBooleanAny,
    "ImageCAS": ImageCAS,
    "Combine_Mask": Combine_Mask,
    "Blur_Mask": Blur_Mask,
    "FloatToInt": FloatToInt,
    "JoinWithDelimiter": JoinWithDelimiter,
    "CR_TextConcatenate": CR_TextConcatenate,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InputCollector": "Standard Input Collector",
    "Image_Filters": "Image Filters",
    "ImageResize": "Image Resize",
    "ImageComposite": "Image Composite",
    "showAnything": "Show Anything",
    "PrepareImageAndMaskForInpaint": "Prepare Image & Mask for Inpaint",
    "OverlayInpaintedLatent": "Overlay Inpainted Latent",
    "OverlayInpaintedImage": "Overlay Inpainted Image",
    "OpenAINode": "OpenAI Node",
    "CSwitchFromAny": "Switch From Any",
    "CSwitchBooleanAny": "Switch Boolean Any",
    "ImageCAS": "Image Contrast Adaptive Sharpening",
    "Combine_Mask": "Combine Mask",
    "Blur_Mask": "Blur Mask",
    "FloatToInt": "Float → Int",
    "JoinWithDelimiter": "Join With Delimiter",
    "CR_TextConcatenate": "CR Text Concatenate",

}

# Merge subdirectory mappings
for class_mappings, display_mappings in subdirectory_mappings:
    # Check for conflicts
    conflicts = set(NODE_CLASS_MAPPINGS.keys()) & set(class_mappings.keys())
    if conflicts and _is_verbose():
        print(f"⚠ Node name conflicts detected: {conflicts}")
        for conflict in conflicts:
            print(f"  Keeping existing node: {conflict}")
    
    # Merge mappings (existing nodes take precedence)
    for key, value in class_mappings.items():
        if key not in NODE_CLASS_MAPPINGS:
            NODE_CLASS_MAPPINGS[key] = value
    
    for key, value in display_mappings.items():
        if key not in NODE_DISPLAY_NAME_MAPPINGS:
            NODE_DISPLAY_NAME_MAPPINGS[key] = value

if _is_verbose():
    print(f"✓ Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

# Set up silent startup environment
try:
    from .silent_startup import setup_silent_environment
    setup_silent_environment()
except ImportError:
    # If silent_startup.py is not available, set defaults manually
    import os
    if "COMFYUI_VERBOSE" not in os.environ:
        os.environ["COMFYUI_VERBOSE"] = "0"
    if os.environ.get("COMFYUI_VERBOSE", "0") == "0":
        os.environ["ULTRALYTICS_SILENT"] = "1"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



## samplers Schedulers

import importlib

# Debug package info (only when COMFY_DEBUG is enabled)
if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
    print(f"(RES4LYF) Debug - Main package name: {__name__}")
    print(f"(RES4LYF) Debug - Main package file: {__file__}")

try:
    from .RES4LYF import sigmas
    if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
        print("(RES4LYF) Debug - sigmas import successful")
except Exception as e:
    if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
        print(f"(RES4LYF) Debug - sigmas import failed: {e}")
        import traceback
        traceback.print_exc()


import torch
from math import *


from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
new_scheduler_name = "bong_tangent"
# Only register if RES4LYF.sigmas imported successfully
if 'sigmas' in globals():
    if new_scheduler_name not in SCHEDULER_HANDLERS:
        bong_tangent_handler = SchedulerHandler(handler=sigmas.bong_tangent_scheduler, use_ms=True)
        SCHEDULER_HANDLERS[new_scheduler_name] = bong_tangent_handler
        SCHEDULER_NAMES.append(new_scheduler_name)
else:
    if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
        print("(RES4LYF) Debug - skipping bong_tangent scheduler registration; sigmas unavailable")


from .RES4LYF.res4lyf import RESplain, init as res4lyf_init

#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

res4lyf_init()

discard_penultimate_sigma_samplers = set((
))


def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # *should* be last item in samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # add custom samplers (presumably) to end of list
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

extra_samplers = {}

extra_samplers = dict(reversed(extra_samplers.items()))

# Call add_samplers to register the custom samplers
add_samplers()


WEB_DIRECTORY = "RES4LYF/web/js"



flags = {
    "zampler"        : False,
    "beta_samplers"  : False,
    "legacy_samplers": False,
}


file_path = os.path.join(os.path.dirname(__file__), "zampler_test_code.txt")
if os.path.exists(file_path):
    try:
        from .RES4LYF.zampler import add_zamplers
        NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
        flags["zampler"] = True
        RESplain("Importing zampler.", debug=True)
    except ImportError:
        try:
            import importlib
            for module_name in ["RES4LYF.zampler", "res4lyf.zampler"]:
                try:
                    zampler_module = importlib.import_module(module_name)
                    add_zamplers = zampler_module.add_zamplers
                    NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
                    flags["zampler"] = True
                    RESplain(f"Importing zampler via {module_name}.", debug=True)
                    break
                except ImportError:
                    continue
            else:
                raise ImportError("Zampler module not found in any path")
        except Exception as e:
            if _is_verbose():
                print(f"(RES4LYF) Failed to import zamplers: {e}")



try:
    from .RES4LYF.beta import add_beta
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
    flags["beta_samplers"] = True
    RESplain("Importing beta samplers.", debug=True)
except ImportError as ie:
    try:
        import importlib
        # Try multiple import strategies due to nested custom_nodes structure
        import_attempts = [
            (".RES4LYF.beta", __name__),
            ("RES4LYF.beta", None),
            (f"{__name__}.RES4LYF.beta", None),
        ]
        for module_name, package in import_attempts:
            try:
                beta_module = importlib.import_module(module_name, package=package)
                add_beta = beta_module.add_beta
                NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
                flags["beta_samplers"] = True
                RESplain(f"Importing beta samplers via {module_name}.", debug=True)
                break
            except ImportError as ie2:
                if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                    print(f"(RES4LYF) Debug - Failed {module_name}: {ie2}")
                continue
        else:
            if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                print(f"(RES4LYF) Debug - Initial beta import error: {ie}")
            raise ImportError("Beta module not found in any path")
    except Exception as e:
        if _is_verbose():
            print(f"(RES4LYF) Failed to import beta samplers: {e}")



try:
    from .RES4LYF.legacy import add_legacy
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
    flags["legacy_samplers"] = True
    RESplain("Importing legacy samplers.", debug=True)
except ImportError as ie:
    try:
        import importlib
        # Try multiple import strategies due to nested custom_nodes structure
        import_attempts = [
            (".RES4LYF.legacy", __name__),
            ("RES4LYF.legacy", None),
            (f"{__name__}.RES4LYF.legacy", None),
        ]
        for module_name, package in import_attempts:
            try:
                legacy_module = importlib.import_module(module_name, package=package)
                add_legacy = legacy_module.add_legacy
                NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
                flags["legacy_samplers"] = True
                RESplain(f"Importing legacy samplers via {module_name}.", debug=True)
                break
            except ImportError as ie2:
                if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                    print(f"(RES4LYF) Debug - Failed {module_name}: {ie2}")
                continue
        else:
            if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                print(f"(RES4LYF) Debug - Initial legacy import error: {ie}")
            raise ImportError("Legacy module not found in any path")
    except Exception as e:
        if _is_verbose():
            print(f"(RES4LYF) Failed to import legacy samplers: {e}")

# Final node registration summary (only in debug)
if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
    print(f"✓ RES4LYF Integration Complete:")
    print(f"  - Zampler samplers: {flags['zampler']}")
    print(f"  - Beta samplers: {flags['beta_samplers']}")
    print(f"  - Legacy samplers: {flags['legacy_samplers']}")
    print(f"  - Total RES4LYF nodes: {len(NODE_CLASS_MAPPINGS)}")

