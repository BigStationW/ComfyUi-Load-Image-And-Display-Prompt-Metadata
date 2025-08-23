from __future__ import annotations
import torch

import os
import sys
import json
import hashlib

from PIL import Image, ImageOps, ImageSequence
import numpy as np

import comfy.model_management
import folder_paths
import node_helpers

class LoadImageX:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive_prompt", "negative_prompt")

    CATEGORY = "testt"
    FUNCTION = "load_image"

    def load_image(self, image, positive_prompt="", negative_prompt=""):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image_rgb = i.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask_np)

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask, positive_prompt, negative_prompt)

    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class OnlyLoadImagesWithMetadata:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["image"])
        
        # Filter files to only include those with metadata
        filtered_files = []
        for file in files:
            if s._has_prompt_metadata(file):
                filtered_files.append(file)
        
        return {
            "required": {
                "image": (sorted(filtered_files), {"image_upload": True})
            },
            "optional": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive_prompt", "negative_prompt")

    CATEGORY = "testt"
    FUNCTION = "load_image"

    @classmethod
    def _has_prompt_metadata(cls, filename):
        """Check if an image file contains prompt metadata"""
        try:
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, filename)
            
            if not os.path.exists(image_path):
                return False
            
            # Read and parse PNG metadata
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Check PNG signature
            png_signature = b'\x89PNG\r\n\x1a\n'
            if not data.startswith(png_signature):
                return False  # Not a PNG file
            
            # Parse PNG chunks to find tEXt or iTXt chunks
            offset = 8  # Skip PNG signature
            while offset < len(data):
                if offset + 8 > len(data):
                    break
                
                length = int.from_bytes(data[offset:offset+4], 'big')
                chunk_type = data[offset+4:offset+8].decode('ascii', errors='ignore')
                
                if chunk_type in ('tEXt', 'iTXt'):
                    chunk_data = data[offset+8:offset+8+length]
                    try:
                        text = chunk_data.decode('utf-8', errors='ignore')
                        null_index = text.find('\0')
                        if null_index != -1:
                            keyword = text[:null_index]
                            value = text[null_index+1:]
                            
                            # Check if this is a prompt metadata chunk
                            if keyword == 'prompt' and value:
                                # Parse the JSON to check for workflow with prompts
                                try:
                                    prompt_data = json.loads(value.replace(': NaN', ': null'))
                                    if cls._workflow_has_prompts(prompt_data):
                                        return True
                                except (json.JSONDecodeError, ValueError):
                                    continue
                    except UnicodeDecodeError:
                        continue
                
                offset += 8 + length + 4  # chunk header + data + CRC
                
                # Break at IEND chunk
                if chunk_type == 'IEND':
                    break
            
            return False
            
        except Exception:
            return False

    @classmethod
    def _workflow_has_prompts(cls, workflow):
        """Check if workflow contains meaningful prompts"""
        if not workflow:
            return False
        
        # Input patterns to look for
        input_patterns = {
            'positive': ['positive', 'conditioning_positive', 'pos'],
            'negative': ['negative', 'conditioning_negative', 'nag_negative', 'neg']
        }
        
        try:
            # Look for nodes with conditioning inputs
            for node_id, node in workflow.items():
                if not isinstance(node, dict) or 'inputs' not in node:
                    continue
                
                node_inputs = node['inputs']
                if not isinstance(node_inputs, dict):
                    continue
                
                # Check for conditioning connections
                for pattern_type, patterns in input_patterns.items():
                    for pattern in patterns:
                        if pattern in node_inputs and isinstance(node_inputs[pattern], list):
                            # Follow the connection to find text
                            connected_node_id = str(node_inputs[pattern][0])
                            if connected_node_id in workflow:
                                connected_node = workflow[connected_node_id]
                                text = cls._extract_text_from_node(connected_node_id, workflow)
                                if text and text.strip():
                                    return True
            
            # Fallback: look for CLIPTextEncode nodes with non-empty text
            for node_id, node in workflow.items():
                if not isinstance(node, dict):
                    continue
                    
                if node.get('class_type') in ['CLIPTextEncode', 'CLIPTextEncodeFlux']:
                    inputs = node.get('inputs', {})
                    if isinstance(inputs, dict):
                        # Check various text fields
                        text_fields = ['text', 'clip_l', 't5xxl', 'prompt']
                        for field in text_fields:
                            if field in inputs:
                                text = inputs[field]
                                if isinstance(text, str) and text.strip():
                                    return True
            
            return False
            
        except Exception:
            return False

    @classmethod
    def _extract_text_from_node(cls, node_id, workflow, visited=None):
        """Extract text from a node, following connections"""
        if visited is None:
            visited = set()
            
        if node_id in visited or node_id not in workflow:
            return ""
        
        visited.add(node_id)
        node = workflow[node_id]
        
        if not isinstance(node, dict):
            return ""
        
        node_type = node.get('class_type', '')
        inputs = node.get('inputs', {})
        
        if not isinstance(inputs, dict):
            return ""
        
        # Handle different node types
        if node_type == 'CLIPTextEncode' and 'text' in inputs:
            text = inputs['text']
            if isinstance(text, list) and text:
                return cls._extract_text_from_node(str(text[0]), workflow, visited)
            elif isinstance(text, str):
                return text
                
        elif node_type == 'CLIPTextEncodeFlux':
            for field in ['clip_l', 't5xxl']:
                if field in inputs:
                    text = inputs[field]
                    if isinstance(text, list) and text:
                        return cls._extract_text_from_node(str(text[0]), workflow, visited)
                    elif isinstance(text, str):
                        return text
                        
        elif node_type == 'String Literal' and 'string' in inputs:
            return str(inputs['string'])
            
        return ""

    def load_image(self, image, positive_prompt="", negative_prompt=""):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image_rgb = i.convert("RGB")
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask_np)

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask, positive_prompt, negative_prompt)

    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODE_CLASS_MAPPINGS = { 
    "LoadImageX": LoadImageX,
    "OnlyLoadImagesWithMetadata": OnlyLoadImagesWithMetadata
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "LoadImageX": "Load Image And Display Prompt Metadata",
    "OnlyLoadImagesWithMetadata": "Only Load Images With Metadata"
}

WEB_DIRECTORY = "./web
