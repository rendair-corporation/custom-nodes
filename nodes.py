import os
import json
import logging
import torch
from typing import Optional, Dict, List, Tuple, Union, Any
from PIL import Image
import numpy as np
import folder_paths
from nodes import LoadImage
from pathlib import Path

version_code = [2, 0, 0]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] InputCollector: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("InputCollector")

logging.info(f"### Loading: Rendair Custom Nodes Pack ({version_str})")


class StyleConfigError(Exception):
    """Custom exception for style configuration errors"""
    pass

class StyleConfig:
    """Configuration class for predefined styles"""

    # Class variable to cache configurations
    _cached_configs = None

    
    @classmethod
    def validate_config_schema(cls, config: dict) -> bool:
        """Validate the basic structure of loaded configuration"""
        try:
            required_top_level = {"checkpoints", "loras", "custom_style"}
            # custom style part is for reference image, if its true, we are also using reference images for style
            
            for style, style_config in config.items():
                missing_keys = required_top_level - set(style_config.keys())
                if missing_keys:
                    logger.error(f"Style '{style}' missing required keys: {missing_keys}")
                    return False
                
                # Validate checkpoints structure
                checkpoints = style_config.get("checkpoints", {})
                for cp_key, cp_value in checkpoints.items():
                    if not isinstance(cp_value, dict) or "name" not in cp_value or "strength" not in cp_value:
                        logger.error(f"Invalid checkpoint configuration in style '{style}': {cp_key}")
                        return False
                
                # Validate LoRAs structure - new format: lora filename as key
                loras = style_config.get("loras", {})
                for lora_filename, lora_config in loras.items():
                    if not isinstance(lora_config, dict) or "min" not in lora_config or "max" not in lora_config:
                        logger.error(f"Invalid LoRA configuration in style '{style}': {lora_filename}")
                        return False
                
# Prompts section removed - no longer needed
                
                # Validate custom_style structure
                custom_style = style_config.get("custom_style", {})
                required_custom_style = {"enabled", "min", "max"}
                if not all(key in custom_style for key in required_custom_style):
                    logger.error(f"Invalid custom_style configuration in style '{style}'")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating config schema: {str(e)}")
            return False

    @classmethod
    def load_style_configs(cls):
        """Load style configurations from JSON file with validation and style name injection"""
        # Return cached configs if available
        if cls._cached_configs is not None:
            return cls._cached_configs
            
        try:
            current_dir = Path(__file__).parent.resolve()
            config_path = current_dir / 'style_configs.json'
            
            if not config_path.exists():
                raise StyleConfigError(f"Style config file not found at {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            # No longer injecting style name into custom_style
            
            if not cls.validate_config_schema(configs):
                raise StyleConfigError("Invalid configuration schema")
            
            logger.info(f"Successfully loaded style configurations from {config_path}")
            
            # Cache the configs
            cls._cached_configs = configs
            return configs
                
        except json.JSONDecodeError as e:
            raise StyleConfigError(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            raise StyleConfigError(f"Error loading style configs: {str(e)}")
    
    @classmethod
    def get_style_configs(cls):
        """Get all available styles and their parameters"""
        if cls._cached_configs is None:
            return cls.load_style_configs()
        return cls._cached_configs
    
    @classmethod
    def get_available_styles(cls):
        """Get list of available style names"""
        try:
            configs = cls.get_style_configs()
            return list(configs.keys())
        except StyleConfigError as e:
            logger.error(f"Error getting available styles: {str(e)}")
            return ["realistic"]  # Return default if there's an error
    
    @classmethod
    def get_style_config(cls, style_name):
        """Get configuration for a specific style"""
        try:
            configs = cls.get_style_configs()
            if style_name not in configs:
                logger.warning(f"Style '{style_name}' not found, using realistic")
                return configs.get("realistic")
            return configs[style_name]
        except StyleConfigError as e:
            logger.error(f"Error getting style config: {str(e)}")
            return None

    @classmethod
    def reload_configs(cls):
        """Force reload of style configurations"""
        try:
            # Clear the cache
            cls._cached_configs = None
            configs = cls.load_style_configs()
            logger.info("Style configurations reloaded successfully")
            return configs
        except Exception as e:
            logger.error(f"Error reloading configurations: {str(e)}")
            raise


class InputCollector:
    """
    A comprehensive input collection and processing node for ComfyUI.
    
    This node handles various types of inputs including images, models, parameters,
    and produces standardized outputs for use in image generation pipelines.
    
    Outputs
    -------
    Images:
        base_image: The main input image (IMAGE)
        reference_image: Reference image for style transfer (IMAGE)
        mask_input: Mask for controlled generation (IMAGE)
        sketch_input: Sketch for controlled generation (IMAGE)
        
    Checkpoints:
        checkpoint_1: Primary model checkpoint (STRING)
        checkpoint_2: Secondary model checkpoint (STRING)
    
    LoRA Outputs (for each i in 1-4):
        style_lora_i_on_off: "ON" or "OFF" based on LoRA activation (STRING)
        style_lora_i: Name of the LoRA model (STRING)
        style_strength_i: Calculated strength for the LoRA (FLOAT)
        
    Style Controls:
        with_custom_style: Enable/disable custom style (BOOLEAN)
        custom_style: Custom style text (STRING)
        custom_style_strength: Strength of custom style (FLOAT)
        
    Personalization:
        with_personalized_custom_style: Enable/disable personalized style (BOOLEAN)
        personalized_custom_style: Personalized style identifier (STRING)
        personalized_custom_style_strength: Strength of personalized style (FLOAT)
        personalized_custom_style_images: Batch of style reference images (IMAGE)
        
    Generation Parameters:
        positive_prompt: Combined positive prompt (STRING)
        negative_prompt: Combined negative prompt (STRING)
        sampler: Selected sampling method (STRING)
        scheduler: Selected scheduler (STRING)
        width: Output image width (INT)
        height: Output image height (INT)
        steps: Number of generation steps (INT)
        
    Strength Controls:
        cfg_creativity: Creativity control strength (FLOAT)
        denoise_color_strength: Color denoising strength (FLOAT)
        controlnet_shape_strength: Shape control strength (FLOAT)
        
    Reference Controls:
        with_reference: Enable/disable reference image (BOOLEAN)
        reference_strength: Reference image influence strength (FLOAT)
        reference_type: Type of reference processing (STRING)
        
    Additional Controls:
        upscale_factor: Image upscaling factor (FLOAT)
        texture strenght: Texture tiling strength (FLOAT)
    
    Version: 1.1.0
    """

    # 1. Class Constants
    VERSION = "1.2.0"
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_STEPS = 40
    
    FUNCTION = "process"
    CATEGORY = "utils"

    # Update the RETURN_TYPES - LoRA outputs removed, handled in prompt suffix
    RETURN_TYPES = (
        "IMAGE",      # base_image
        "IMAGE",      # reference_image
        "BOOLEAN",    # mask_provided
        "IMAGE",      # mask_input
        "BOOLEAN",    # sketch_provided
        "IMAGE",      # sketch_input
        "IMAGE",      # sketch_mask
        "BOOLEAN",    # insert_image_provided
        "IMAGE",      # insert_image
        "IMAGE",      # insert_image_mask
        "STRING",     # checkpoint_1
        "STRING",     # checkpoint_2
        "BOOLEAN",    # with_custom_style
        "STRING",     # custom_style
        "FLOAT",      # custom_style_strength   
        "BOOLEAN",    # with_personalized_custom_style
        "STRING",     # personalized_custom_style
        "FLOAT",      # personalized_custom_style_strength
        "IMAGE",      # personalized_custom_style_images
        "STRING",     # positive_prompt
        "STRING",     # negative_prompt
        "STRING",     # sampler
        "STRING",     # scheduler
        "INT",        # width
        "INT",        # height
        "INT",        # steps
        "FLOAT",      # cfg_creativity
        "FLOAT",      # denoise_color_strength
        "FLOAT",      # controlnet_shape_strength
        "BOOLEAN",    # with_reference
        "FLOAT",      # reference_strength
        "STRING",     # reference_type
        "INT",        # upscale_factor
        "FLOAT",      # texture strenght
        "IMAGE",      # style_reference_images
        "INT",        # upscale_batch_size
        "INT",        # k_steps
        "INT",        # k_denoise
    )

    RETURN_NAMES = (
        "base_image",
        "reference_image",
        "mask_provided",
        "mask_input",
        "sketch_provided",
        "sketch_input",
        "sketch_mask",
        "insert_image_provided",
        "insert_image",
        "insert_image_mask",
        "checkpoint_1",
        "checkpoint_2",
        "with_custom_style",
        "custom_style",
        "custom_style_strength",
        "with_personalized_custom_style",
        "personalized_custom_style",
        "personalized_custom_style_strength",
        "personalized_custom_style_images",
        "positive_prompt",
        "negative_prompt",
        "sampler",
        "scheduler",
        "width",
        "height",
        "steps",
        "cfg_creativity",
        "denoise_color_strength",
        "controlnet_shape_strength",
        "with_reference",
        "reference_strength",
        "reference_type",
        "upscale_factor",
        "texture strenght",
        "style_reference_images",
        "upscale_batch_size",
        "k_steps",
        "k_denoise",
    )
    

    # Aspect ratio configurations
    ASPECT_RATIOS = {
        ("1:1", "horizontal"): (1024, 1024),
        ("1:1", "vertical"): (1024, 1024),
        ("5:4", "horizontal"): (1280, 1024),
        ("5:4", "vertical"): (1024, 1280),
        ("16:9", "horizontal"): (1820, 1024),
        ("16:9", "vertical"): (1024, 1820),
        ("3:2", "horizontal"): (1536, 1024),
        ("3:2", "vertical"): (1024, 1536)
    }

    def __init__(self):
        self.image_loader = LoadImage()


    @classmethod
    def INPUT_TYPES(s):
        """Define input parameters and their types."""
        try:
            input_dir = os.path.join(os.path.dirname(folder_paths.__file__), "input")
            input_files = []
            if os.path.exists(input_dir):
                input_files = [f for f in os.listdir(input_dir) 
                             if os.path.isfile(os.path.join(input_dir, f))]
            
            return {
                "required": {
                    
                    # Single style selector
                    "chosen_style": (StyleConfig.get_available_styles(),),
                    # Main style strength control
                    "style_strength": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),

                    # Strength Parameters
                    "creativity": ("FLOAT", {"default": 70, "min": 0, "max": 100, "step": 1}),
                    "creativity_min": ("FLOAT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                    "creativity_max": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 1}),
        
                    "color_strength": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                    "color_strength_min": ("FLOAT", {"default": 0, "min": 0, "max": 50, "step": 0.01}),
                    "color_strength_max": ("FLOAT", {"default": 1, "min": 0, "max": 50, "step": 0.01}),
        
                    "shape_strength": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                    "shape_strength_min": ("FLOAT", {"default": 0, "min": 0, "max": 2, "step": 0.01}),
                    "shape_strength_max": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.01}),

                    # Reference toggle
                    "with_reference": (["OFF", "ON"],),
                    "reference_strength": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                    "reference_strength_min": ("FLOAT", {"default": 0, "min": 0, "max": 2, "step": 0.01}),
                    "reference_strength_max": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.01}),
                    "reference_type": (["linear", "style transfer", "strong style transfer", "style transfer precise"],),

                    # Upscale controls
                    "upscale_factor": (["2K", "4K", "6K", "8K"],),
                    "texture": ("FLOAT", {"default": 10, "min": 0, "max": 100, "step": 0.1}),
                    "texture_min": ("FLOAT", {"default": 4, "min": 0, "max": 100, "step": 0.1}),
                    "texture_max": ("FLOAT", {"default": 80, "min": 0, "max": 100, "step": 0.1}),
                    
                    # Personalized Custom style
                    "with_personalized_custom_style": (["OFF", "ON"],),
                    "personalized_custom_style_strength": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                    "personalized_custom_style_min": ("FLOAT", {"default": 0, "min": 0, "max": 2, "step": 0.01}),
                    "personalized_custom_style_max": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.01}),
                    "personalized_custom_style_image_folder_path": ("STRING", {
                        "multiline": False, 
                        "default": "test-images"
                    }),

                    # Base Models and LoRAs with OFF option
                    "checkpoint_1": (["OFF"] + folder_paths.get_filename_list("checkpoints"),),
                    "checkpoint_2": (["OFF"] + folder_paths.get_filename_list("checkpoints"),),
                    
                    # Image dimensions control
                    "aspect_ratio": (["1:1", "5:4", "3:2", "16:9"],),
                    "orientation": (["horizontal", "vertical"],),
                    
                    # Generation Parameters
                    "sampler_name": (["random", "res_2m", "deis_2m", "res_2s", "dpmpp_2m", "uni_pc"],),
                    "scheduler_name": (["random", "bong_tangent", "beta", "beta57", "sgm_uniform", "normal"],),
                    
                    # Generation Settings
                    "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                    
                },
                "optional": {
                    # Image Inputs
                    "base_image": (["OFF"] + input_files, {"image_upload": True}),
                    #"reference_image": (["OFF"] + input_files, {"image_upload": True}),
                    # New: load a batch of reference images from a subfolder inside the input directory
                    "reference_image_folder_path": ("STRING", {"multiline": False, "default": ""}),
                    "mask_input": (["OFF"] + input_files, {"image_upload": True}),
                    "sketch_input": (["OFF"] + input_files, {"image_upload": True}),
                    "insert_image": (["OFF"] + input_files, {"image_upload": True}),
                    "sketch_mask": (["OFF"] + input_files, {"image_upload": True}),
                    "insert_image_mask": (["OFF"] + input_files, {"image_upload": True}),
                    
                    # Simple prompts
                    "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                    "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
    
                    # Notes
                    "Notes": ("STRING", {"multiline": True, "default": ""})
                }
            }
        except Exception as e:
            logger.error(f"Error in INPUT_TYPES: {str(e)}")
            raise




    def get_sampler_scheduler_name(self, sampler_name: str, scheduler_name: str) -> tuple:
        """Get the sampler and scheduler names with proper coupling logic."""
        import random
        
        # Define the valid sampler-scheduler couples
        valid_couples = [
            ("res_2m", "bong_tangent"),
            ("res_2m", "sgm_uniform"),
            ("deis_2m", "normal"),
            ("deis_2m", "sgm_uniform"),
            ("res_2s", "sgm_uniform"),
            ("res_2s", "beta"),
            ("res_2s", "bong_tangent"),
            ("res_2s", "beta57"),
            ("dpmpp_2m", "beta57"),
            ("dpmpp_2m", "beta"),
            ("dpmpp_2m", "bong_tangent"),
            ("dpmpp_2m", "sgm_uniform"),
            ("uni_pc", "bong_tangent"),
            ("uni_pc", "beta"),
            ("uni_pc", "beta57")
        ]
        
        # Handle random sampler selection
        if sampler_name == "random" and scheduler_name == "random":
            # Both random: select a random couple
            selected_sampler, selected_scheduler = random.choice(valid_couples)
        elif sampler_name == "random":
            # Only sampler is random: find valid samplers for the given scheduler
            valid_samplers = [sampler for sampler, scheduler in valid_couples if scheduler == scheduler_name]
            if valid_samplers:
                selected_sampler = random.choice(valid_samplers)
                selected_scheduler = scheduler_name
            else:
                # Fallback to a random couple if no valid sampler found for the scheduler
                logger.warning(f"No valid sampler found for scheduler '{scheduler_name}', using random couple")
                selected_sampler, selected_scheduler = random.choice(valid_couples)
        elif scheduler_name == "random":
            # Only scheduler is random: find valid schedulers for the given sampler
            valid_schedulers = [scheduler for sampler, scheduler in valid_couples if sampler == sampler_name]
            if valid_schedulers:
                selected_scheduler = random.choice(valid_schedulers)
                selected_sampler = sampler_name
            else:
                # Fallback to a random couple if no valid scheduler found for the sampler
                logger.warning(f"No valid scheduler found for sampler '{sampler_name}', using random couple")
                selected_sampler, selected_scheduler = random.choice(valid_couples)
        else:
            # Both specified: use as-is
            selected_sampler = sampler_name
            selected_scheduler = scheduler_name
            
        return selected_sampler, selected_scheduler


    def process_image(self, image_path: str) -> Optional[Any]:
        """Process and load an image."""
        try:
            if image_path in [None, "", "None", "OFF"]:
                logger.info(f"Image processing: OFF")
                return None
            
            image = self.image_loader.load_image(image_path)[0]
            logger.info(f"Image processing: Loaded {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def map_value(self, value: float, min_val: float, max_val: float, param_name: str = "") -> float:
        """Map percentage values to target range."""
        try:
            if value is None:
                value = 0
            mapped_value = min_val + (max_val - min_val) * (value / 100.0)
            logger.info(f"{param_name}: {value}% -> {mapped_value:.3f} (range: {min_val:.3f} to {max_val:.3f})")
            return mapped_value
        except Exception as e:
            logger.error(f"Error mapping value for {param_name}: {str(e)}")
            return min_val
    
    def generate_lora_prompt_suffix(self, loras_config: dict, style_strength: float) -> str:
        """Generate LoRA prompt suffix in the format <lora:name:strength>"""
        try:
            lora_tags = []
            
            for lora_filename, lora_config in loras_config.items():
                # Extract the LoRA name without the .safetensors extension
                lora_name = lora_filename.replace('.safetensors', '')
                
                # Calculate strength using the style's min/max values and global style strength
                lora_strength = self.map_value(
                    style_strength,
                    lora_config["min"],
                    lora_config["max"],
                    f"lora_{lora_name}_strength"
                )
                
                # Create the LoRA tag: <lora:name:strength>
                lora_tag = f"<lora:{lora_name}:{lora_strength:.2f}>"
                lora_tags.append(lora_tag)
                
                logger.info(f"Generated LoRA tag: {lora_tag}")
            
            # Join all LoRA tags with commas and spaces
            lora_suffix = ", ".join(lora_tags)
            logger.info(f"Complete LoRA suffix: {lora_suffix}")
            
            return lora_suffix
            
        except Exception as e:
            logger.error(f"Error generating LoRA prompt suffix: {str(e)}")
            return ""
    
    def load_images_from_folder(self, folder_path: str) -> Optional[Any]:
        try:
            input_folder = os.path.join("/workspace/ComfyUI/input", folder_path)
            
            logger.info(f"Looking for images in: {input_folder}")
            if not os.path.exists(input_folder):
                logger.error(f"Folder not found: {input_folder}")
                return torch.zeros((1, 1024, 1024, 3))
            
            MAX_SIZE = 1024  # Maximum dimension for any side
            TARGET_SIZE = 1024  # Final square size after cropping
            
            # Find all valid image files in the folder
            image_files = []
            supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
            
            for filename in sorted(os.listdir(input_folder)):
                if filename.lower().endswith(supported_formats):
                    image_files.append(os.path.join(input_folder, filename))
            
            # Count valid image files
            num_files = len(image_files)
            logger.info(f"Found {num_files} image files in folder")
            
            if num_files == 0:
                logger.info("No valid images found in folder")
                return torch.zeros((1, 1024, 1024, 3))
            
            # SINGLE IMAGE CASE: Just load and resize if needed, but keep aspect ratio
            if num_files == 1:
                file_path = image_files[0]
                logger.info(f"Single image mode: Loading image {os.path.basename(file_path)} without cropping")
                try:
                    image = self.image_loader.load_image(file_path)[0]
                    _, img_height, img_width, _ = image.shape
                    
                    # Only resize if largest dimension exceeds MAX_SIZE
                    max_dim = max(img_width, img_height)
                    if max_dim > MAX_SIZE:
                        # Calculate new dimensions that preserve aspect ratio
                        if img_width >= img_height:
                            # Width is the larger dimension
                            new_width = MAX_SIZE
                            new_height = int(img_height * (MAX_SIZE / img_width))
                        else:
                            # Height is the larger dimension
                            new_height = MAX_SIZE
                            new_width = int(img_width * (MAX_SIZE / img_height))
                        
                        logger.info(f"Resizing single image from {img_width}x{img_height} to {new_width}x{new_height}")
                        
                        # Resize the image
                        image_for_resize = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
                        resized = torch.nn.functional.interpolate(
                            image_for_resize, 
                            size=(new_height, new_width),
                            mode='bilinear',
                            align_corners=False
                        )
                        image = resized.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
                    
                    logger.info(f"Successfully loaded single image with dimensions {image.shape}")
                    return image
                    
                except Exception as e:
                    logger.error(f"Error loading single image: {str(e)}")
                    return torch.zeros((1, 1024, 1024, 3))
            
            # MULTIPLE IMAGES CASE: Process all images, resize and crop to ensure uniform size
            processed_images = []
            
            for file_path in image_files:
                filename = os.path.basename(file_path)
                logger.info(f"Multiple image mode: Processing {filename}")
                try:
                    # Load the image
                    image = self.image_loader.load_image(file_path)[0]
                    _, img_height, img_width, _ = image.shape
                    
                    # Step 1: First resize if needed while preserving aspect ratio
                    max_dim = max(img_width, img_height)
                    if max_dim > MAX_SIZE:
                        # Calculate new dimensions that preserve aspect ratio
                        if img_width >= img_height:
                            # Width is the larger dimension
                            new_width = MAX_SIZE
                            new_height = int(img_height * (MAX_SIZE / img_width))
                        else:
                            # Height is the larger dimension
                            new_height = MAX_SIZE
                            new_width = int(img_width * (MAX_SIZE / img_height))
                        
                        logger.info(f"Resizing {filename} from {img_width}x{img_height} to {new_width}x{new_height}")
                        
                        # Resize the image
                        image_for_resize = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
                        resized = torch.nn.functional.interpolate(
                            image_for_resize, 
                            size=(new_height, new_width),
                            mode='bilinear',
                            align_corners=False
                        )
                        image = resized.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
                    
                    # Step 2: Crop to square if necessary
                    _, img_height, img_width, _ = image.shape  # Get the new dimensions after resize
                    
                    if img_height != img_width:
                        logger.info(f"Cropping {filename} to square {TARGET_SIZE}x{TARGET_SIZE}")
                        
                        # Center crop to square
                        if img_width > img_height:
                            # Image is wider than tall, crop width
                            start_w = (img_width - img_height) // 2
                            image = image[:, :, start_w:start_w+img_height, :]
                        else:
                            # Image is taller than wide, crop height
                            start_h = (img_height - img_width) // 2
                            image = image[:, start_h:start_h+img_width, :, :]
                    
                    # Step 3: Final resize to exact TARGET_SIZE if not already that size
                    _, img_height, img_width, _ = image.shape
                    if img_height != TARGET_SIZE or img_width != TARGET_SIZE:
                        image_for_resize = image.permute(0, 3, 1, 2)
                        resized = torch.nn.functional.interpolate(
                            image_for_resize, 
                            size=(TARGET_SIZE, TARGET_SIZE),
                            mode='bilinear',
                            align_corners=False
                        )
                        image = resized.permute(0, 2, 3, 1)
                    
                    processed_images.append(image)
                    logger.info(f"Successfully processed image: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing image {filename}: {str(e)}")
                    continue
            
            if not processed_images:
                logger.info("No images were successfully processed")
                return torch.zeros((1, 1024, 1024, 3))
            
            # Concatenate all images (they're all the same size now)
            result = torch.cat(processed_images, dim=0) if len(processed_images) > 1 else processed_images[0]
            logger.info(f"Final tensor batch shape: {result.shape}")
            return result
                
        except Exception as e:
            logger.error(f"Error loading images from folder {folder_path}: {str(e)}")
            return torch.zeros((1, 1024, 1024, 3))

    def find_style_reference_images(self, style_name: str) -> Optional[Any]:
        try:
            if not style_name:
                return torch.zeros((1, 512, 512, 3))

            base_path = os.path.dirname(os.path.realpath(__file__))
            refs_path = os.path.join(base_path, 'references')

            # Use fixed folder name previously used by the style selector: rendair_custom_styles
            ref_image_dir = os.path.join(refs_path, 'rendair_custom_styles', style_name)

            if not os.path.exists(ref_image_dir):
                # Ensure directory exists to guide user
                os.makedirs(ref_image_dir, exist_ok=True)
                return torch.zeros((1, 512, 512, 3))

            images: List[torch.Tensor] = []
            supported_formats = ('.jpg', '.jpeg', '.png', '.webp')

            for filename in sorted(os.listdir(ref_image_dir)):
                if not filename.lower().endswith(supported_formats):
                    continue
                image_path = os.path.join(ref_image_dir, filename)
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                    if len(img_tensor.shape) == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    images.append(img_tensor)
                except Exception:
                    continue

            if not images:
                return torch.zeros((1, 512, 512, 3))

            if len(images) == 1:
                return images[0]

            # Pad to max dims and concatenate
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            padded_images: List[torch.Tensor] = []
            for img in images:
                if img.shape[1] != max_h or img.shape[2] != max_w:
                    pad_h = max_h - img.shape[1]
                    pad_w = max_w - img.shape[2]
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    padded = torch.nn.functional.pad(
                        img,
                        (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=0.5,
                    )
                    padded_images.append(padded)
                else:
                    padded_images.append(img)
            return torch.cat(padded_images, dim=0)
        except Exception:
            return torch.zeros((1, 512, 512, 3))


    def process(self, **kwargs):
        """Main processing method for all inputs."""
        try:
            logger.info("="*50)
            logger.info("Starting input processing")
            logger.info("="*50)

            results = {}
            
            # Get style configuration based on chosen_style
            chosen_style = kwargs.get("chosen_style", "watercolor")
            style_config = StyleConfig.get_style_config(chosen_style)
            logger.info(f"Processing style: {chosen_style}")

            # Image Processing
            logger.info("Processing Images:")
            for img_key in ["base_image", "reference_image", "mask_input", "sketch_input", "insert_image", "sketch_mask", "insert_image_mask"]:
                img = self.process_image(kwargs.get(img_key))
                results[img_key] = img
                logger.info(f"  - {img_key}: {'Loaded' if img is not None else 'Not provided'}")

            # Set boolean flags for provided inputs based on successfully loaded images
            results["mask_provided"] = results["mask_input"] is not None
            results["sketch_provided"] = results["sketch_input"] is not None
            results["insert_image_provided"] = results["insert_image"] is not None
            
            # If a reference folder path is provided, load all images in that subfolder
            # from /workspace/ComfyUI/input and use the batch as the reference_image output
            try:
                ref_folder_path = kwargs.get("reference_image_folder_path", "")
                if isinstance(ref_folder_path, str) and ref_folder_path.strip():
                    ref_batch = self.load_images_from_folder(ref_folder_path.strip())
                    if ref_batch is not None:
                        results["reference_image"] = ref_batch
                        logger.info("Loaded batch reference images from folder: %s", ref_folder_path)
            except Exception as e:
                logger.error("Error loading reference images from folder: %s", str(e))

            # Log the automatically determined provided flags
            logger.info(f"Mask provided: {results['mask_provided']}")
            logger.info(f"Sketch provided: {results['sketch_provided']}")
            logger.info(f"Insert image provided: {results['insert_image_provided']}")

            # Process checkpoints from style configuration
            logger.info("Processing Checkpoints:")
            checkpoint_config = style_config.get("checkpoints", {})
            
            # Process checkpoint_1
            checkpoint1_info = checkpoint_config.get("checkpoint_1", {"name": "OFF", "strength": 0.0})
            results["checkpoint_1"] = checkpoint1_info["name"]
            logger.info(f"  - Checkpoint 1: {checkpoint1_info['name']} (strength: {checkpoint1_info['strength']})")

            # Process checkpoint_2
            checkpoint2_info = checkpoint_config.get("checkpoint_2", {"name": "OFF", "strength": 0.0})
            results["checkpoint_2"] = checkpoint2_info["name"]
            logger.info(f"  - Checkpoint 2: {checkpoint2_info['name']} (strength: {checkpoint2_info['strength']})")

            # Process personalized custom style images folder
            personalized_images = torch.zeros((1, 512, 512, 3))  # Default tensor instead of None
            if kwargs.get("with_personalized_custom_style", "OFF") == "ON":
                folder_path = kwargs.get("personalized_custom_style_image_folder_path", "")
                if folder_path:
                    try:
                        loaded_images = self.load_images_from_folder(folder_path)
                        if loaded_images is not None:
                            personalized_images = loaded_images
                            logger.info("Successfully loaded personalized style images")
                    except Exception as e:
                        logger.error(f"Error loading personalized style images: {str(e)}")
            
            results["personalized_custom_style_images"] = personalized_images

            # Generate LoRA prompt suffix from style configuration
            logger.info("Processing Style Configuration:")
            
            # Generate LoRA suffix for prompt
            loras_config = style_config.get("loras", {})
            lora_suffix = self.generate_lora_prompt_suffix(loras_config, kwargs.get("style_strength", 50))
            
            # Process custom style from style config  
            custom_config = style_config["custom_style"]
            results["with_custom_style"] = custom_config["enabled"]
            results["custom_style"] = chosen_style  # Use the chosen style name as custom style text
            results["custom_style_strength"] = self.map_value(
                kwargs.get("style_strength", 50),
                custom_config["min"],
                custom_config["max"],
                "custom_style_strength"
            )
            
            logger.info(f"Custom style: {'enabled' if custom_config['enabled'] else 'disabled'}")
            if custom_config["enabled"]:
                logger.info(f"  - Text: {chosen_style}")
                logger.info(f"  - Strength: {results['custom_style_strength']:.3f}")

            # Process personalized custom style
            with_personalized_custom_style = kwargs.get("with_personalized_custom_style", "OFF") == "ON"
            results["with_personalized_custom_style"] = with_personalized_custom_style
            if with_personalized_custom_style:
                results["personalized_custom_style_strength"] = self.map_value(
                    kwargs.get("personalized_custom_style_strength", 50),
                    kwargs.get("personalized_custom_style_min", 0.0),
                    kwargs.get("personalized_custom_style_max", 1.0),
                    "personalized_custom_style_strength"
                )
                # Set the folder name as the personalized style identifier
                folder_path = kwargs.get("personalized_custom_style_image_folder_path", "")
                results["personalized_custom_style"] = os.path.basename(folder_path) if folder_path else ""
            else:
                results["personalized_custom_style_strength"] = 0
                results["personalized_custom_style"] = ""  # Empty string when disabled

            # Process prompts with style-specific prefixes and suffixes
            try:
                base_positive_prompt = kwargs.get("positive_prompt", "").strip()
                base_negative_prompt = kwargs.get("negative_prompt", "").strip()

                # Get pre/post prompts from style config if present
                pre_pos = style_config.get("pre_positive_prompt", "").strip()
                post_pos = style_config.get("post_positive_prompt", "").strip()
                pre_neg = style_config.get("pre_negative_prompt", "").strip()
                post_neg = style_config.get("post_negative_prompt", "").strip()

                # Positive: pre + user + post + lora_suffix
                positive_parts: List[str] = []
                if pre_pos:
                    positive_parts.append(pre_pos)
                if base_positive_prompt:
                    positive_parts.append(base_positive_prompt)
                if post_pos:
                    positive_parts.append(post_pos)
                if lora_suffix:
                    positive_parts.append(lora_suffix)
                results["positive_prompt"] = ", ".join([p for p in positive_parts if p])

                # Negative: pre + user + post
                negative_parts: List[str] = []
                if pre_neg:
                    negative_parts.append(pre_neg)
                if base_negative_prompt:
                    negative_parts.append(base_negative_prompt)
                if post_neg:
                    negative_parts.append(post_neg)
                results["negative_prompt"] = ", ".join([p for p in negative_parts if p])

                logger.info(f"Final Positive Prompt: {results['positive_prompt']}")
                logger.info(f"Final Negative Prompt: {results['negative_prompt']}")

            except Exception as e:
                logger.error(f"Error processing prompts: {str(e)}")
                raise
        
            # Generation Parameters
            sampler, scheduler = self.get_sampler_scheduler_name(kwargs.get("sampler_name"), kwargs.get("scheduler_name"))
            results["sampler"]   = sampler
            results["scheduler"] = scheduler
            logger.info(f"Sampler: {results['sampler']}")
            logger.info(f"Scheduler: {results['scheduler']}")

            # Dimensions
            aspect_ratio = kwargs.get("aspect_ratio", "1:1")
            orientation = kwargs.get("orientation", "horizontal")
            width, height = self.ASPECT_RATIOS.get((aspect_ratio, orientation), (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
            results["width"] = width
            results["height"] = height

            # Steps - ensure it's an integer
            results["steps"] = int(kwargs.get("steps", self.DEFAULT_STEPS))

            # Process all strength parameters
            strength_params = [
                ("creativity", "cfg_creativity", "creativity_min", "creativity_max"),
                ("color_strength", "denoise_color_strength", "color_strength_min", "color_strength_max"),
                ("shape_strength", "controlnet_shape_strength", "shape_strength_min", "shape_strength_max")
            ]

            for base_name, output_name, min_name, max_name in strength_params:
                value = self.map_value(
                    kwargs.get(base_name),
                    kwargs.get(min_name),
                    kwargs.get(max_name),
                    output_name
                )
                results[output_name] = value

            # Process reference parameters
            results["with_reference"] = kwargs.get("with_reference", "OFF") == "ON"
            if results["with_reference"]:
                results["reference_strength"] = self.map_value(
                    kwargs.get("reference_strength"),
                    kwargs.get("reference_strength_min"),
                    kwargs.get("reference_strength_max"),
                    "reference_strength"
                )
            else:
                results["reference_strength"] = 0
            
            results["reference_type"] = kwargs.get("reference_type")

            # Process upscale factor
            upscale_map = {"2K": 2048, "4K": 4096, "6K": 6144, "8K": 8192}
            results["upscale_factor"] = upscale_map.get(kwargs.get("upscale_factor", "2K"), 2048)

            # Process upscale batch size based on selected upscale factor option
            batch_size_map = {"2K": 16, "4K": 7, "6K": 6, "8K": 6}
            selected_upscale_key = kwargs.get("upscale_factor", "2K")
            results["upscale_batch_size"] = int(batch_size_map.get(selected_upscale_key, 16))

            # Process texture
            results["texture strenght"] = self.map_value(
                kwargs.get("texture"),
                kwargs.get("texture_min"),
                kwargs.get("texture_max"),
                "texture strenght"
            )

            logger.info("="*50)
            # Style reference images loaded from references/rendair_custom_styles/<style_name>
            results["style_reference_images"] = self.find_style_reference_images(chosen_style)

            logger.info("Processing complete")
            logger.info("="*50)

            # Convert results dictionary to a tuple based on RETURN_NAMES order
            # Compute k_steps and k_denoise based on styles/ref toggles and chosen_style
            try:
                any_active = bool(results.get("with_custom_style")) \
                    or bool(results.get("with_personalized_custom_style")) \
                    or bool(results.get("with_reference")) \
                    or (str(chosen_style).lower() != "realistic")
                if any_active:
                    results["k_steps"] = 4
                    results["k_denoise"] = 7
                else:
                    results["k_steps"] = 10
                    results["k_denoise"] = 15
            except Exception:
                results["k_steps"] = 10
                results["k_denoise"] = 15

            output_tuple = tuple(results.get(name, None) for name in self.RETURN_NAMES)
            return output_tuple

        except Exception as e:
            logger.error(f"Error in process method: {str(e)}")
            raise


class FloatToInt:
    """Simple utility node: rounds a FLOAT to the nearest INT."""

    CATEGORY = "utils"
    FUNCTION = "convert"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "value": ("FLOAT", {"default": 0.0, "min": -1_000_000_000.0, "max": 1_000_000_000.0, "step": 0.01}),
                "value_2": ("FLOAT", {"default": 0.0, "min": -1_000_000_000.0, "max": 1_000_000_000.0, "step": 0.01}),
            }
        }

    def convert(self, value: float = 0.0, value_2: float = 0.0):
        try:
            total_value = float(value) + (float(value_2) if value_2 is not None else 0.0)
            rounded = int(round(total_value))
        except Exception:
            rounded = 0
        return (rounded,)

NODE_CLASS_MAPPINGS = {
    "InputCollector": InputCollector,
    "FloatToInt": FloatToInt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InputCollector": "Standard Input Collector",
    "FloatToInt": "Float → Int"
}