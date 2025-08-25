import sys
import logging

# Add logger definition
logger = logging.getLogger("SwitchAny")

FLOAT = ("FLOAT", {"default": 1,
                   "min": -sys.float_info.max,
                   "max": sys.float_info.max,
                   "step": 0.01})

BOOLEAN = ("BOOLEAN", {"default": True})
BOOLEAN_FALSE = ("BOOLEAN", {"default": False})

INT = ("INT", {"default": 1,
               "min": -sys.maxsize,
               "max": sys.maxsize,
               "step": 1})

STRING = ("STRING", {"default": ""})

STRING_ML = ("STRING", {"multiline": True, "default": ""})

STRING_WIDGET = ("STRING", {"forceInput": True})

JSON_WIDGET = ("JSON", {"forceInput": True})

METADATA_RAW = ("METADATA_RAW", {"forceInput": True})

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class CSwitchBooleanAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": (any, {"lazy": True}),
                "on_false": (any, {"lazy": True}),
                "boolean": BOOLEAN,
            }
        }

    # Fix the CATEGORY definition - line 54
    CATEGORY = "utils"
    RETURN_TYPES = (any,)

    FUNCTION = "execute"

    def check_lazy_status(self, on_true=None, on_false=None, boolean=True):
        needed = "on_true" if boolean else "on_false"
        return [needed]

    def execute(self, on_true, on_false, boolean=True):
        logger.debug("Any switch: " + str(boolean))

        if boolean:
            return (on_true,)
        else:
            return (on_false,)


class CSwitchFromAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (any, ),
                "boolean": BOOLEAN,
            }
        }

    CATEGORY = "utils"
    RETURN_TYPES = (any, any,)
    RETURN_NAMES = ("on_true", "on_false",)

    FUNCTION = "execute"

    def execute(self, any,boolean=True):
        logger.debug("Any switch: " + str(boolean))

        if boolean:
            return any, None
        else:
            return None, any