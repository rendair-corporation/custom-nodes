import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Utility functions for tensor-PIL conversion
def tensor2pil(tensor):
    """Convert a tensor to PIL Image"""
    # Handle single image tensor [H, W, C] or batch tensor [1, H, W, C]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Ensure tensor is in range [0, 1] and convert to [0, 255]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = (tensor * 255).to(torch.uint8)
    
    # Convert to numpy and then to PIL
    array = tensor.cpu().numpy()
    return Image.fromarray(array)

def pil2tensor(pil_image):
    """Convert PIL Image to tensor"""
    # Convert PIL to numpy array
    array = np.array(pil_image)
    
    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(array).float() / 255.0
    
    # Add batch dimension [H, W, C] -> [1, H, W, C]
    return tensor.unsqueeze(0)

# SIMPLE IMAGE ADJUST

class Image_Filters:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "gaussian_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "edge_enhance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_enhance": (["false", "true"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_filters"

    CATEGORY = "Image_Filter"

    def image_filters(self, image, brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):


        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None

                # Apply NP Adjustments
                if brightness > 0.0 or brightness < 0.0:
                    # Apply brightness
                    img = np.clip(img + brightness, 0.0, 1.0)

                if contrast > 1.0 or contrast < 1.0:
                    # Apply contrast
                    img = np.clip(img * contrast, 0.0, 1.0)

                # Apply PIL Adjustments
                if saturation > 1.0 or saturation < 1.0:
                    # PIL Image
                    pil_image = tensor2pil(img)
                    # Apply saturation
                    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

                if sharpness > 1.0 or sharpness < 1.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply sharpness
                    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

                if blur > 0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply blur
                    for _ in range(blur):
                        pil_image = pil_image.filter(ImageFilter.BLUR)

                if gaussian_blur > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply Gaussian blur
                    pil_image = pil_image.filter(
                        ImageFilter.GaussianBlur(radius=gaussian_blur))

                if edge_enhance > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Edge Enhancement
                    edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    # Blend Mask
                    blend_mask = Image.new(
                        mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                    # Composite Original and Enhanced Version
                    pil_image = Image.composite(
                        edge_enhanced_img, pil_image, blend_mask)
                    # Clean-up
                    del blend_mask, edge_enhanced_img

                if detail_enhance == "true":
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = pil_image.filter(ImageFilter.DETAIL)

                # Output image
                out_image = (pil2tensor(pil_image) if pil_image else img.unsqueeze(0))

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:

            pil_image = None
            img = image

            # Apply NP Adjustments
            if brightness > 0.0 or brightness < 0.0:
                # Apply brightness
                img = np.clip(img + brightness, 0.0, 1.0)

            if contrast > 1.0 or contrast < 1.0:
                # Apply contrast
                img = np.clip(img * contrast, 0.0, 1.0)

            # Apply PIL Adjustments
            if saturation > 1.0 or saturation < 1.0:
                # PIL Image
                pil_image = tensor2pil(img)
                # Apply saturation
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness > 1.0 or sharpness < 1.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply sharpness
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if blur > 0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply blur
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)

            if gaussian_blur > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply Gaussian blur
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Edge Enhancement
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                # Blend Mask
                blend_mask = Image.new(
                    mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                # Composite Original and Enhanced Version
                pil_image = Image.composite(
                    edge_enhanced_img, pil_image, blend_mask)
                # Clean-up
                del blend_mask, edge_enhanced_img

            if detail_enhance == "true":
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = pil_image.filter(ImageFilter.DETAIL)

            # Output image
            out_image = (pil2tensor(pil_image) if pil_image else img)

            tensors = out_image

        return (tensors, )

# MASK COMBINE

class Combine_Mask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "mask_a": ("MASK",),
                        "mask_b": ("MASK",),
                    },
                    "optional": {
                        "mask_c": ("MASK",),
                        "mask_d": ("MASK",),
                        "mask_e": ("MASK",),
                        "mask_f": ("MASK",),
                    }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "combine_masks"

    def combine_masks(self, mask_a, mask_b, mask_c=None, mask_d=None, mask_e=None, mask_f=None):
        # Gather all masks in a list
        masks = [m for m in [mask_a, mask_b, mask_c, mask_d, mask_e, mask_f] if m is not None]

        # Skip any masks that are the known "empty" shape [1, 64, 64] from "Preview" etc
        # (You can also use a sum-of-pixels check, or other logic.)
        valid_masks = [m for m in masks if m.shape != (1, 64, 64)]
        # cstr(f"mask shapes: ... `{valid_masks}`").msg.print()

        # If no valid masks, decide on a fallback
        if len(valid_masks) == 0:
            # Could return a zeroed-out mask, or just return mask_a, or raise a warning
            # Return mask_a so we don't break the graph
            return (mask_a, )

        # If there is exactly one valid mask, no combine needed
        if len(valid_masks) == 1:
            return (valid_masks[0], )

        # Otherwise stack, sum, clamp
        combined_mask = torch.sum(torch.stack(valid_masks, dim=0), dim=0)
        combined_mask = torch.clamp(combined_mask, 0, 1)  # Keep values in 0..1

        return (combined_mask,)


class Blur_Mask:
    """Blur a mask using a separable Gaussian kernel.

    Inputs:
    - mask: (B, H, W) tensor in [0, 1]
    - blur: odd kernel size (pixels). 0 or 1 returns the original mask.
    """

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur": ("INT", {"default": 7, "min": 0, "max": 8191, "step": 1}),
            }
        }

    @staticmethod
    def _make_odd(value: int) -> int:
        v = int(value)
        return v if v % 2 == 1 else v + 1

    @staticmethod
    def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype):
        half = (kernel_size - 1) / 2.0
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
        kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        return kernel_1d

    def apply(self, mask: torch.Tensor, blur: int):
        # Accept empty/placeholder masks as-is
        if mask is None:
            return (mask,)

        # If no blur requested or very small kernel, return original
        if blur <= 1:
            return (mask,)

        kernel_size = self._make_odd(blur)

        # Derive sigma from kernel size for a pleasant falloff
        sigma = max(0.1, float(kernel_size) / 3.0)

        # Ensure tensor has batch dimension (B, H, W)
        if mask.dim() == 2:
            mask_bhw = mask.unsqueeze(0)
        else:
            mask_bhw = mask

        device = mask_bhw.device
        dtype = mask_bhw.dtype

        # Prepare separable kernels
        k1d = self._gaussian_kernel_1d(kernel_size, sigma, device, dtype)
        kx = k1d.view(1, 1, 1, kernel_size)
        ky = k1d.view(1, 1, kernel_size, 1)

        x = mask_bhw.unsqueeze(1)  # (B,1,H,W)
        x = F.conv2d(x, ky, padding=(kernel_size // 2, 0))
        x = F.conv2d(x, kx, padding=(0, kernel_size // 2))
        x = x.squeeze(1).clamp(0.0, 1.0)

        return (x,)


# Node registration for ComfyUI
# just for documentation
NODE_CLASS_MAPPINGS = {
    "Image_Filters": Image_Filters,
    "Combine_Mask": Combine_Mask,
    "Blur_Mask": Blur_Mask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Filters": "Image Filters",
    "Combine_Mask": "Combine Mask",
    "Blur_Mask": "Blur Mask",
}