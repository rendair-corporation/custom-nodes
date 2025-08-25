import json

# AlwaysEqualProxy class for universal input type handling
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

# Define any_type using AlwaysEqualProxy for ComfyUI wildcard input
any_type = AlwaysEqualProxy("*")

class showAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (any_type, {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "util"

    def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):

        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if type(val) is str:
                        values.append(val)
                    elif type(val) is list:
                        values = val
                    else:
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass

        if not extra_pnginfo:
            pass
        elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
            pass
        else:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]
        
        # Format text for UI display
        if isinstance(values, list):
            if len(values) == 1:
                display_text = str(values[0])
                return {"ui": {"text": [display_text]}, "result": (values[0],)}
            else:
                display_text = "\n".join(str(v) for v in values)
                return {"ui": {"text": [display_text]}, "result": (values,)}
        else:
            display_text = str(values) if values else ""
            return {"ui": {"text": [display_text]}, "result": (values,)}