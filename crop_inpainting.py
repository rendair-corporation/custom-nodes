import numpy as np
from PIL import Image, ImageFilter, ImageOps
from enum import Enum
import cv2
import torch
from typing import Dict

import os
import io
import sys
import base64
import importlib
import importlib.metadata
import subprocess
from packaging import version
from packaging.specifiers import SpecifierSet
from PIL import Image
import sys
import copy
import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("ArtVenture")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("[%(name)s] - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Configure logger
loglevel = logging.INFO
logger.setLevel(loglevel)



def get_crop_region(mask: np.ndarray, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""

    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left - pad, 0)),
        int(max(crop_top - pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h)),
    )


def expand_crop_region(crop_region: np.ndarray, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128.
    """

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new("RGBA", (image.width, image.height))

    image_masked = Image.new("RGBa", (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))

    image_masked = image_masked.convert("RGBa")

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")



class ResizeMode(Enum):
    RESIZE = 0  # just resize
    RESIZE_TO_FILL = 1  # crop and resize
    RESIZE_TO_FIT = 2  # resize and fill


def resize_image(im: Image.Image, width: int, height: int, resize_mode=ResizeMode.RESIZE_TO_FIT):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    def resize(im: Image.Image, w, h):
        return im.resize((w, h), resample=Image.LANCZOS)

    if resize_mode == ResizeMode.RESIZE:
        res = resize(im, width, height)

    elif resize_mode == ResizeMode.RESIZE_TO_FILL:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(
                    resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(
                    resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                    box=(fill_width + src_w, 0),
                )

    return res


def flatten_image(im: Image.Image, bgcolor="#ffffff"):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""

    if im.mode == "RGBA":
        background = Image.new("RGBA", im.size, bgcolor)
        background.paste(im, mask=im)
        im = background

    return im.convert("RGB")


## Utils



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def ensure_package(package, required_version=None, install_package_name=None):
    # Try to import the package
    try:
        module = importlib.import_module(package)
    except ImportError:
        logger.info(f"Package {package} is not installed. Installing now...")
        install_command = _construct_pip_command(install_package_name or package, required_version)
        subprocess.check_call(install_command)
    else:
        # If a specific version is required, check the version
        if required_version:
            try:
                installed_version = importlib.metadata.version(package)
                
                # Parse version specifier (e.g., ">=1.1.1", "==1.1.1", "<=1.1.1")
                if any(op in required_version for op in ['>=', '<=', '==', '!=', '>', '<', '~=']):
                    spec = SpecifierSet(required_version)
                    if installed_version not in spec:
                        logger.info(
                            f"Package {package} version constraint not satisfied (installed: {installed_version}, required: {required_version}). Installing now..."
                        )
                        install_command = _construct_pip_command(install_package_name or package, required_version)
                        subprocess.check_call(install_command)
                else:
                    # Fallback to simple version comparison for backwards compatibility
                    if version.parse(installed_version) < version.parse(required_version):
                        logger.info(
                            f"Package {package} is outdated (installed: {installed_version}, required: {required_version}). Upgrading now..."
                        )
                        install_command = _construct_pip_command(install_package_name or package, required_version)
                        subprocess.check_call(install_command)
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"Package {package} version information not found. Installing required version {required_version}...")
                install_command = _construct_pip_command(install_package_name or package, required_version)
                subprocess.check_call(install_command)


def _construct_pip_command(package_name, version=None):
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    # Include the version in the package name if specified
    if version:
        package_name = f"{package_name}=={version}"

    return pip_install + [package_name]


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    nested_keys = name_string.split(".")
    value = dict_inst

    for key in nested_keys:
        # Handle array indexing
        if key.startswith("[") and key.endswith("]"):
            try:
                index = int(key[1:-1])
                if not isinstance(value, (list, tuple)) or index >= len(value):
                    return default
                value = value[index]
            except (ValueError, TypeError):
                return default
        else:
            if not isinstance(value, dict):
                return default
            value = value.get(key, None)

        if value is None:
            return default

    return value


def set_dict_attribute(dict_inst: dict, name_string: str, value):
    """
    Set an attribute to a dictionary using dot notation.
    If the attribute does not already exist, it will create a nested dictionary.

    Parameters:
        - dict_inst: the dictionary instance to set the attribute
        - name_string: the attribute name in dot notation (ex: 'attributes[1].name')
        - value: the value to set for the attribute

    Returns:
        None
    """
    # Split the attribute names by dot
    name_list = name_string.split(".")

    # Traverse the dictionary and create a nested dictionary if necessary
    current_dict = dict_inst
    for name in name_list[:-1]:
        is_array = name.endswith("]")
        if is_array:
            open_bracket_index = name.index("[")
            idx = int(name[open_bracket_index + 1 : -1])
            name = name[:open_bracket_index]

        if name not in current_dict:
            current_dict[name] = [] if is_array else {}

        current_dict = current_dict[name]
        if is_array:
            while len(current_dict) <= idx:
                current_dict.append({})
            current_dict = current_dict[idx]

    # Set the final attribute to its value
    name = name_list[-1]
    if name.endswith("]"):
        open_bracket_index = name.index("[")
        idx = int(name[open_bracket_index + 1 : -1])
        name = name[:open_bracket_index]

        if name not in current_dict:
            current_dict[name] = []

        while len(current_dict[name]) <= idx:
            current_dict[name].append(None)

        current_dict[name][idx] = value
    else:
        current_dict[name] = value


def is_junction(src: str) -> bool:
    import subprocess

    child = subprocess.Popen('fsutil reparsepoint query "{}"'.format(src), stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    return rc == 0


def load_module(module_path, module_name=None):
    import importlib.util

    if module_name is None:
        module_name = os.path.basename(module_path)
        if os.path.isdir(module_path):
            module_path = os.path.join(module_path, "__init__.py")

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def tensor2bytes(image: torch.Tensor) -> bytes:
    return tensor2pil(image).tobytes()


def pil2base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



# node
class PrepareImageAndMaskForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64}),
                "inpaint_masked": ("BOOLEAN", {"default": False}),
                "mask_padding": ("INT", {"default": 32, "min": 0, "max": 1024}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048}),
            },
            "optional": {
                "controlnet_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "CROP_REGION", "IMAGE")
    RETURN_NAMES = ("inpaint_image", "inpaint_mask", "overlay_image", "crop_region", "controlnet_image")
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "prepare"

    def prepare(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_blur: int,
        inpaint_masked: bool,
        mask_padding: int,
        width: int,
        height: int,
        controlnet_image: torch.Tensor = None,
    ):
        if image.shape[0] != mask.shape[0]:
            raise ValueError("image and mask must have same batch size")

        if controlnet_image is not None and image.shape[0] != controlnet_image.shape[0]:
            raise ValueError("image and controlnet_image must have same batch size")

        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError("image and mask must have same dimensions")

        # These are only used if inpaint_masked is True
        out_width, out_height = width, height
        if inpaint_masked and out_width == 0 and out_height == 0:
            out_height, out_width = image.shape[1:3]

        source_height, source_width = image.shape[1:3]

        images = []
        masks = []
        overlay_images = []
        crop_regions = []
        processed_controlnet_images = []

        for idx, (img, msk) in enumerate(zip(image, mask)):
            np_mask: np.ndarray = msk.cpu().numpy()

            if mask_blur > 0:
                kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)

            pil_mask = numpy2pil(np_mask, "L")
            pil_img = tensor2pil(img)
            
            # --- LOGIC SEPARATION ---

            if inpaint_masked:
                # --- MODE 1: CROP AND RESIZE ---
                crop_region = get_crop_region(np_mask, mask_padding)
                crop_region = expand_crop_region(crop_region, out_width, out_height, source_width, source_height)
                
                cropped_img = pil_img.crop(crop_region)
                cropped_mask = pil_mask.crop(crop_region)

                final_pil_img = resize_image(cropped_img, out_width, out_height, ResizeMode.RESIZE_TO_FIT)
                final_pil_mask = resize_image(cropped_mask, out_width, out_height, ResizeMode.RESIZE_TO_FIT).convert("L")

                if controlnet_image is not None:
                    pil_cimg = tensor2pil(controlnet_image[idx])
                    cn_source_width, cn_source_height = pil_cimg.size
                    scale_x = cn_source_width / source_width
                    scale_y = cn_source_height / source_height
                    
                    cn_target_width = int(out_width * scale_x)
                    cn_target_height = int(out_height * scale_y)
                    
                    x1, y1, x2, y2 = crop_region
                    cn_crop_region = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                    cropped_cn_img = pil_cimg.crop(cn_crop_region)
                    final_cn_img = resize_image(cropped_cn_img, cn_target_width, cn_target_height, ResizeMode.RESIZE_TO_FIT)
                    processed_controlnet_images.append(pil2tensor(final_cn_img))

            else:
                # --- MODE 2: PASS-THROUGH (NO RESIZING) ---
                final_pil_img = pil_img
                final_pil_mask = pil_mask # Already blurred if requested
                crop_region = (0, 0, source_width, source_height)

                if controlnet_image is not None:
                    # Simply pass the original controlnet image through
                    final_cn_img = tensor2pil(controlnet_image[idx])
                    processed_controlnet_images.append(pil2tensor(final_cn_img))

            # --- COMMON LOGIC FOR BOTH MODES ---
            
            # The overlay/preview should always be based on the original full-size image
            image_masked = Image.new("RGBa", (pil_img.width, pil_img.height))
            # The mask used here is the potentially blurred one, but before any cropping/resizing
            image_masked.paste(pil_img.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(pil_mask))
            overlay_images.append(pil2tensor(image_masked.convert("RGBA")))

            images.append(pil2tensor(final_pil_img))
            masks.append(pil2tensor(final_pil_mask))
            crop_regions.append(torch.tensor(crop_region, dtype=torch.int64))


        if processed_controlnet_images:
            final_controlnet_tensor = torch.cat(processed_controlnet_images, dim=0)
        else:
            # If no controlnet image is provided, create a black 64x64 placeholder
            batch_size = image.shape[0]
            final_controlnet_tensor = torch.zeros((batch_size, 64, 64, 3), dtype=torch.float32, device=image.device)

        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(overlay_images, dim=0),
            torch.stack(crop_regions, dim=0),
            final_controlnet_tensor,
        )


class OverlayInpaintedLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("LATENT",),
                "inpainted": ("LATENT",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, original: Dict, inpainted: Dict, mask: torch.Tensor):
        s_original: torch.Tensor = original["samples"]
        s_inpainted: torch.Tensor = inpainted["samples"]

        if s_original.shape[0] != s_inpainted.shape[0]:
            raise ValueError("original and inpainted must have same batch size")

        if s_original.shape[0] != mask.shape[0]:
            raise ValueError("original and mask must have same batch size")

        overlays = []

        for org, inp, msk in zip(s_original, s_inpainted, mask):
            latmask = tensor2pil(msk.unsqueeze(0), "L").convert("RGB").resize((org.shape[2], org.shape[1]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            msk = torch.asarray(1.0 - latmask)
            nmask = torch.asarray(latmask)

            overlayed = inp * nmask + org * msk
            overlays.append(overlayed)

        samples = torch.stack(overlays)
        return ({"samples": samples},)


class OverlayInpaintedImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpainted": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "crop_region": ("CROP_REGION",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, inpainted: torch.Tensor, overlay_image: torch.Tensor, crop_region: torch.Tensor):
        if inpainted.shape[0] != overlay_image.shape[0]:
            raise ValueError("inpainted and overlay_image must have same batch size")
        if inpainted.shape[0] != crop_region.shape[0]:
            raise ValueError("inpainted and crop_region must have same batch size")

        images = []
        for image, overlay, region in zip(inpainted, overlay_image, crop_region):
            image = tensor2pil(image.unsqueeze(0))
            overlay = tensor2pil(overlay.unsqueeze(0), mode="RGBA")

            x1, y1, x2, y2 = region.tolist()
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                pass
            else:
                base_image = Image.new("RGBA", (overlay.width, overlay.height))
                image = resize_image(image, x2 - x1, y2 - y1, ResizeMode.RESIZE_TO_FILL)
                base_image.paste(image, (x1, y1))
                image = base_image

            image = image.convert("RGBA")
            image.alpha_composite(overlay)
            image = image.convert("RGB")

            images.append(pil2tensor(image))

        return (torch.cat(images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "PrepareImageAndMaskForInpaint": PrepareImageAndMaskForInpaint,
    "OverlayInpaintedLatent": OverlayInpaintedLatent,
    "OverlayInpaintedImage": OverlayInpaintedImage,
}

# remove lama model was here, add it back in case of need
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareImageAndMaskForInpaint": "Prepare Image & Mask for Inpaint",
    "OverlayInpaintedLatent": "Overlay Inpainted Latent",
    "OverlayInpaintedImage": "Overlay Inpainted Image",
}