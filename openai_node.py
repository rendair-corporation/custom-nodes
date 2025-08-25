import torch
from server import PromptServer
import base64
from PIL import Image
import io
import json
from openai import AsyncOpenAI, OpenAI
import os
import asyncio
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class OpenAINode:
    @classmethod
    def read_config(cls):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('api_key', '')
        except Exception as e:
            print(f"Warning: Could not read config.json: {str(e)}")
            return ''

    def __init__(self):
        # Prefer config.json, fallback to env
        self.api_key = self.read_config() or os.getenv('OPENAI_API_KEY', '')
        if not self.api_key:
            print("Warning: No API key found in config.json or environment variable")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["text", "vision"], ),
                "model": (["gpt-5o", "gpt-5o-mini", "gpt-5o-nano", "gpt-4o", "gpt-4o-mini"], ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "detail_level": (["auto", "low", "high"], {
                    "default": "auto", 
                    "description": "Controls image processing detail"
                }),
                "fix_people": ("BOOLEAN", {"default": False, "description": "If true, combine all responses into DSC-SIZE/SEP formatted single string"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "process_request"
    CATEGORY = "AI/OpenAI"

    def encode_image_tensor(self, image_tensor):
        start_time = time.time()
        
        # Squeeze any singleton dimensions and ensure proper shape
        if len(image_tensor.shape) == 4:
            # Remove batch dimensions if they're size 1
            while len(image_tensor.shape) > 3 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
        
        # Convert to CPU and numpy
        image_np = image_tensor.cpu().numpy()
        
        # Handle different data types
        if str(image_np.dtype) in ['uint8', '|u1']:
            # Already in 0-255 range
            image_np = image_np.astype('uint8')
        else:
            # Assume float in 0-1 range, convert to 0-255
            image_np = (image_np * 255).astype('uint8')
        
        # Convert to PIL Image
        image = Image.fromarray(image_np)
        
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        target_size = 768  # Max size for efficient processing
        if width > target_size or height > target_size:
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))
            print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height))
        
        # Convert to base64 with optimized settings
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print(f"Image encoding took {time.time() - start_time:.2f} seconds")
        return encoded

    async def process_text_request(self, client, model, prompt):
        """Process text-only requests using async client"""
        api_start = time.time()
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        api_time = time.time() - api_start
        print(f"Text API request time: {api_time:.2f}s")
        return completion.choices[0].message.content

    async def process_single_image(self, client, model, prompt, image_tensor, detail_level, index=None):
        total_start = time.time()
        
        # Time the image encoding
        encode_start = time.time()
        base64_image = self.encode_image_tensor(image_tensor)
        encode_time = time.time() - encode_start
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail_level
                }
            }
        ]
        
        # Time the API request - now uses the selected model parameter
        api_start = time.time()
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        api_time = time.time() - api_start
        
        total_time = time.time() - total_start
        print(f"Image {index if index is not None else 0} processing breakdown:")
        print(f"  - Encoding time: {encode_time:.2f}s")
        print(f"  - API request time: {api_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        
        return completion.choices[0].message.content

    async def process_batch(self, client, model, prompt, image_batch, detail_level):
        batch_start = time.time()
        num_images = image_batch.shape[0]
        print(f"Starting batch processing of {num_images} images")
        
        tasks = []
        for i in range(num_images):
            single_image = image_batch[i]
            task = self.process_single_image(client, model, prompt, single_image, detail_level, index=i)
            tasks.append(task)
        
        # Process all images concurrently
        responses = await asyncio.gather(*tasks)
        
        batch_time = time.time() - batch_start
        print(f"Total batch processing time: {batch_time:.2f}s")
        print(f"Batch processing completed: {len(responses)} responses generated")
        
        # Debug: print first few characters of each response
        for i, response in enumerate(responses):
            print(f"Response {i}: {response[:50]}..." if len(response) > 50 else f"Response {i}: {response}")
        
        return responses

    async def run_async_request(self, model, prompt, image=None, detail_level="auto"):
        """Main async function that handles both text and vision requests automatically"""
        client = AsyncOpenAI(api_key=self.api_key)
        
        # Automatically detect mode based on image presence
        if image is None:
            # Text mode
            print("Auto-detected: Text mode (no image provided)")
            response = await self.process_text_request(client, model, prompt)
            return [response]
        else:
            # Vision mode
            print("Auto-detected: Vision mode (image provided)")
            print(f"Image tensor shape: {image.shape}")
            print(f"Image tensor type: {type(image)}")
            
            # Check if we have a batch of images (4D tensor) or single image (3D tensor)
            if len(image.shape) == 4 and image.shape[0] > 1:
                # Batch of images
                print(f"Processing batch of {image.shape[0]} images")
                responses = await self.process_batch(client, model, prompt, image, detail_level)
                print(f"Batch processing returned {len(responses)} responses")
            else:
                # Single image (could be 3D or 4D with batch size 1)
                if len(image.shape) == 4:
                    # 4D tensor with batch size 1, squeeze the batch dimension
                    single_image = image[0]
                    print("Processing single image (squeezed from batch dimension)")
                else:
                    # 3D tensor
                    single_image = image
                    print("Processing single image (3D tensor)")
                
                response = await self.process_single_image(client, model, prompt, single_image, detail_level)
                responses = [response]
                print(f"Single image processing returned 1 response")
            
            print(f"Final responses count: {len(responses)}")
            return responses

    def run_in_thread(self, model, prompt, image=None, detail_level="auto"):
        """Run async code in a separate thread to avoid event loop conflicts"""
        def run_async():
            return asyncio.run(self.run_async_request(model, prompt, image, detail_level))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            return future.result()

    def process_request(self, mode, model, prompt, image=None, detail_level="auto", fix_people=False, unique_id=None):
        start_time = time.time()
        try:
            if not self.api_key:
                return (["Error: No API key found in config.json. Please add your OpenAI API key to the config file."],)

            # Unwrap list-wrapped inputs when INPUT_IS_LIST = True
            mode = mode[0] if isinstance(mode, list) and mode else mode
            model = model[0] if isinstance(model, list) and model else model
            prompt = prompt[0] if isinstance(prompt, list) and prompt else prompt
            detail_level = detail_level[0] if isinstance(detail_level, list) and detail_level else detail_level
            fix_people = fix_people[0] if isinstance(fix_people, list) and fix_people else fix_people

            # Handle multiple images from list input
            if isinstance(image, list):
                all_responses = []
                print(f"Processing {len(image)} images from list input")
                
                if mode == "text":
                    # Process all text requests synchronously
                    print("Using synchronous client for text mode")
                    client = OpenAI(api_key=self.api_key)
                    for i, img in enumerate(image):
                        if img is None:
                            continue
                        print(f"Processing text request {i+1}/{len(image)}")
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response = completion.choices[0].message.content
                        all_responses.append(response)
                else:
                    # Process all vision requests in a single event loop
                    print("Using asynchronous client for vision mode")
                    
                    async def process_all_images():
                        async_client = AsyncOpenAI(api_key=self.api_key)
                        tasks = []
                        for i, img in enumerate(image):
                            if img is None:
                                continue
                            print(f"Scheduling image {i+1}/{len(image)}")
                            task = self.process_single_image(async_client, model, prompt, img, detail_level, index=i)
                            tasks.append(task)
                        
                        responses = await asyncio.gather(*tasks)
                        return responses
                    
                    with ThreadPoolExecutor() as executor:
                        loop = asyncio.new_event_loop()
                        try:
                            all_responses = executor.submit(
                                lambda: loop.run_until_complete(process_all_images())
                            ).result()
                        finally:
                            # Ensure the loop is properly closed
                            loop.close()
                
                # Format output based on fix_people flag
                if fix_people:
                    print(f"fix_people=True: combining {len(all_responses)} responses from list input")
                    # Add pre and post prompts to each response
                    pre_prompt = "RAW portrait photo, looking away, Caucasian, "
                    post_prompt = ", high quality, 8k, detailed portrait photo"
                    formatted_responses = [f"{pre_prompt}{r}{post_prompt}" for r in all_responses]
                    combined = "[DSC-SIZE]\n" + "\n".join([f"{r} [SEP]" for r in formatted_responses])
                    print(f"Combined result: {combined[:100]}...")
                    total_time = time.time() - start_time
                    print(f"Total request processing time: {total_time:.2f}s")
                    return ([combined],)
                else:
                    total_time = time.time() - start_time
                    print(f"Total request processing time: {total_time:.2f}s")
                    return (all_responses,)

            print(f"Processing request in {mode} mode")
            if mode == "text":
                # For text mode, use sync client
                print("Using synchronous client for text mode")
                client = OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses = [completion.choices[0].message.content]
                # Format output if fix_people is enabled
                if fix_people:
                    print(f"fix_people=True (text mode): combining {len(responses)} responses")
                    # Add pre and post prompts to each response
                    pre_prompt = "RAW portrait photo, looking away, Caucasian, "
                    post_prompt = ", high quality, 8k, detailed portrait photo"
                    formatted_responses = [f"{pre_prompt}{r}{post_prompt}" for r in responses]
                    combined = "[DSC-SIZE]\n" + "\n".join([f"{r} [SEP]" for r in formatted_responses])
                    print(f"Combined result: {combined[:100]}...")
                    return ([combined],)
                return (responses,)

            # For vision mode
            if image is None:
                return (["Error: Image is required for vision mode"],)

            # Use async client for vision requests
            print("Using asynchronous client for vision mode")
            async_client = AsyncOpenAI(api_key=self.api_key)

            # If `image` might be a Python list, normalize into a list of 3D tensors
            images_list = None
            if isinstance(image, list):
                images_list = []
                for item in image:
                    if item is None:
                        continue
                    if torch.is_tensor(item):
                        if len(item.shape) == 4:
                            if item.shape[0] <= 1:
                                images_list.append(item[0] if item.shape[0] == 1 else item.squeeze(0))
                            else:
                                for j in range(item.shape[0]):
                                    images_list.append(item[j])
                        elif len(item.shape) == 3:
                            images_list.append(item)
                if not images_list:
                    images_list = None

            # If images_list exists, try to stack them to a batch if shapes match
            if images_list is not None:
                try:
                    shapes = [tuple(t.shape) for t in images_list]
                    if len(set(shapes)) == 1:
                        image = torch.stack(images_list, dim=0)
                        print(f"Stacked {len(images_list)} images into a batch tensor: {image.shape}")
                except Exception:
                    pass

            # Create event loop in a new thread
            with ThreadPoolExecutor() as executor:
                if torch.is_tensor(image) and len(image.shape) == 4:
                    # Batch of images
                    print(f"Processing batch of {image.shape[0]} images")
                    loop = asyncio.new_event_loop()
                    responses = executor.submit(
                        lambda: loop.run_until_complete(
                            self.process_batch(async_client, model, prompt, image, detail_level)
                        )
                    ).result()
                elif images_list is not None and len(images_list) > 1:
                    # Heterogeneous list of images (varying shapes)
                    print(f"Processing list of {len(images_list)} images with varying shapes")
                    async def run_list_images():
                        tasks = [
                            self.process_single_image(async_client, model, prompt, img, detail_level, index=i)
                            for i, img in enumerate(images_list)
                        ]
                        return await asyncio.gather(*tasks)
                    loop = asyncio.new_event_loop()
                    responses = executor.submit(
                        lambda: loop.run_until_complete(run_list_images())
                    ).result()
                else:
                    # Single image
                    print("Processing single image")
                    loop = asyncio.new_event_loop()
                    response = executor.submit(
                        lambda: loop.run_until_complete(
                            self.process_single_image(async_client, model, prompt, image, detail_level)
                        )
                    ).result()
                    responses = [response]
                # If fix_people: combine into a single formatted string
                if fix_people:
                    print(f"fix_people=True: combining {len(responses)} responses")
                    # Add pre and post prompts to each response
                    pre_prompt = "RAW portrait photo, looking away, Caucasian, "
                    post_prompt = ", high quality, 8k, detailed portrait photo"
                    formatted_responses = [f"{pre_prompt}{r}{post_prompt}" for r in responses]
                    combined = "[DSC-SIZE]\n" + "\n".join([f"{r} [SEP]" for r in formatted_responses])
                    print(f"Combined result: {combined[:100]}...")
                    responses = [combined]
                
                total_time = time.time() - start_time
                print(f"Total request processing time: {total_time:.2f}s")
                return (responses,)

        except Exception as e:
            error_msg = str(e)
            total_time = time.time() - start_time
            print(f"Error occurred after {total_time:.2f}s: {error_msg}")
            PromptServer.instance.send_sync("openai.error", {"error": error_msg})
            return ([f"Error: {error_msg}"],)