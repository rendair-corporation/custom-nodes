import os
import importlib

if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
    print("(RES4LYF) Debug - Starting RES4LYF __init__.py")

try:
    from . import sigmas
    if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
        print("(RES4LYF) Debug - RES4LYF sigmas import successful")
except Exception as e:
    if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
        print(f"(RES4LYF) Debug - RES4LYF sigmas import failed: {e}")
        import traceback
        traceback.print_exc()


import torch
from math import *


try:
    from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
    new_scheduler_name = "bong_tangent"
    # Only register if sigmas import was successful
    if 'sigmas' in globals():
        if new_scheduler_name not in SCHEDULER_HANDLERS:
            bong_tangent_handler = SchedulerHandler(handler=sigmas.bong_tangent_scheduler, use_ms=True)
            SCHEDULER_HANDLERS[new_scheduler_name] = bong_tangent_handler
            SCHEDULER_NAMES.append(new_scheduler_name)
    else:
        if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
            print("(RES4LYF) Debug - skipping bong_tangent registration; sigmas unavailable")
except ImportError:
    # ComfyUI not available during import, will be handled later
    pass


from .res4lyf import RESplain, init
#
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

init()

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

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
    
}


WEB_DIRECTORY = "RES4LYF/web/js"



flags = {
    "zampler"        : False,
    "beta_samplers"  : False,
    "legacy_samplers": False,
}


file_path = os.path.join(os.path.dirname(__file__), "zampler_test_code.txt")
if os.path.exists(file_path):
    try:
        from .zampler import add_zamplers
        NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
        flags["zampler"] = True
        RESplain("Importing zampler.")
    except ImportError:
        try:
            import importlib
            for module_name in ["RES4LYF.zampler", "res4lyf.zampler"]:
                try:
                    zampler_module = importlib.import_module(module_name)
                    add_zamplers = zampler_module.add_zamplers
                    NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
                    flags["zampler"] = True
                    RESplain(f"Importing zampler via {module_name}.")
                    break
                except ImportError:
                    continue
            else:
                raise ImportError("Zampler module not found in any path")
        except Exception as e:
            if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
                print(f"(RES4LYF) Failed to import zamplers: {e}")



try:
    from .beta import add_beta
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
    flags["beta_samplers"] = True
    RESplain("Importing beta samplers.", debug=True)
except ImportError:
    try:
        import importlib
        for module_name in ["RES4LYF.beta", "res4lyf.beta"]:
            try:
                beta_module = importlib.import_module(module_name)
                add_beta = beta_module.add_beta
                NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
                flags["beta_samplers"] = True
                RESplain(f"Importing beta samplers via {module_name}.")
                break
            except ImportError:
                continue
        else:
            raise ImportError("Beta module not found in any path")
    except Exception as e:
        if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
            print(f"(RES4LYF) Failed to import beta samplers: {e}")



try:
    from .legacy import add_legacy
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
    flags["legacy_samplers"] = True
    RESplain("Importing legacy samplers.", debug=True)
except ImportError:
    try:
        import importlib
        for module_name in ["RES4LYF.legacy", "res4lyf.legacy"]:
            try:
                legacy_module = importlib.import_module(module_name)
                add_legacy = legacy_module.add_legacy
                NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers)
                flags["legacy_samplers"] = True
                RESplain(f"Importing legacy samplers via {module_name}.")
                break
            except ImportError:
                continue
        else:
            raise ImportError("Legacy module not found in any path")
    except Exception as e:
        if os.environ.get('COMFY_DEBUG', '').lower() in ('1', 'true'):
            print(f"(RES4LYF) Failed to import legacy samplers: {e}")


add_samplers()

WEB_DIRECTORY = "web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]




