from typing import Iterable, List, Any

# Reuse pack's wildcard any type
from .show_anything import any_type


def _flatten_to_strings(values: Any) -> List[str]:
    """Flatten nested iterables into a flat list of strings.

    - None values are skipped
    - Non-iterables are converted to str
    - Strings are treated as atomic
    """
    flat: List[str] = []

    def _recurse(item: Any):
        if item is None:
            return
        # Treat strings as atomic values
        if isinstance(item, str):
            flat.append(item)
            return
        # Handle lists/tuples (common for ComfyUI list inputs)
        if isinstance(item, (list, tuple)):
            for sub in item:
                _recurse(sub)
            return
        # Fallback: stringify
        flat.append(str(item))

    _recurse(values)
    return flat


class JoinWithDelimiter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": (any_type,),
                "delimiter": ([(
                    "newline"), ("comma"), ("backslash"), ("space")],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "♾️Mixlab/Text"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)

    def run(self, text_list, delimiter):
        # ComfyUI provides list-wrapped inputs when INPUT_IS_LIST is True
        delimiter_value = delimiter[0] if isinstance(delimiter, list) and delimiter else delimiter

        if delimiter_value == "newline":
            sep = "\n"
        elif delimiter_value == "comma":
            sep = ","
        elif delimiter_value == "backslash":
            sep = "\\"
        elif delimiter_value == "space":
            sep = " "
        else:
            sep = str(delimiter_value) if delimiter_value is not None else ""

        parts = _flatten_to_strings(text_list)
        joined = sep.join(parts)
        return (joined,)


class CR_TextConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text1": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "text2": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
                "separator": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    # Keep first output as any to allow flexible downstream connections, and a help URL as second
    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("STRING", "show_help")
    FUNCTION = "concat_text"
    CATEGORY = "Comfyroll/Utils/Text"

    def concat_text(self, text1: str = "", text2: str = "", separator: str = ""):
        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/List-Nodes#cr-save-text-to-file"
        result = f"{text1}{separator}{text2}"
        return (result, show_help)


