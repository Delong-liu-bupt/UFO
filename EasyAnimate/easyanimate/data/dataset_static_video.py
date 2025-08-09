import csv
import json
import os
import random
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Import get_random_mask from the original dataset module
def get_random_mask(size):
    """Generate random mask for inpainting"""
    b, f, c, h, w = size
    mask = torch.zeros((b, f, 1, h, w))
    
    for i in range(b):
        # Random mask generation logic
        mask_h = random.randint(h//4, h//2)
        mask_w = random.randint(w//4, w//2)
        mask_y = random.randint(0, h - mask_h)
        mask_x = random.randint(0, w - mask_w)
        
        mask[i, :, 0, mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 1
    
    return mask


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    """Context manager for VideoReader to ensure proper resource cleanup"""
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr


def resize_frame(frame, target_short_side):
    """Resize frame while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    if h < w:
        new_h = target_short_side
        new_w = int(w * (target_short_side / h))
    else:
        new_w = target_short_side
        new_h = int(h * (target_short_side / w))
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame


class StaticVideoDataset(Dataset):
    """
    Dataset for creating static videos from either video files or images.
    
    For video files: extracts the middle frame and repeats it to create a static video
    For image files: repeats the image to create a static video
    
    This is useful for training models to generate consistent static content.
    """
    
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512, 
        video_sample_stride=4, 
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
    ):
        """
        Initialize the StaticVideoDataset
        
        Args:
            ann_path: Path to annotation file (CSV or JSON)
            data_root: Root directory for data files
            video_sample_size: Target size for video processing
            video_sample_stride: Stride for video sampling (not used for static videos)
            video_sample_n_frames: Number of frames to generate for static video
            image_sample_size: Target size for image processing
            video_repeat: Number of times to repeat video entries in dataset
            text_drop_ratio: Probability of dropping text descriptions
            enable_bucket: Whether to enable bucket sampling
            video_length_drop_start: Start ratio for video length dropping
            video_length_drop_end: End ratio for video length dropping
            enable_inpaint: Whether to enable inpainting mode
        """
        # Load annotations from files
        print(f"Loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # Balance dataset between images and videos
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        
        # Repeat video entries if specified
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"Dataset scale: {self.length}")
        
        # Dataset configuration
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video processing parameters
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        
        # Image processing parameters
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        
        # Transform for video frames
        self.video_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size),
        ])
        
        # Transform for images
        self.image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
        ])

    def get_batch(self, idx):
        """Get a single batch item from the dataset"""
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        
        if data_type == 'video':
            sample = self._process_video(data_info)
        else:
            sample = self._process_image(data_info)
        
        # Return in the format expected by the original dataset interface
        return sample["pixel_values"], sample["text"], sample["data_type"]
    
    def _process_video(self, data_info):
        """
        Process video file to create static video using middle frame
        
        Args:
            data_info: Dictionary containing video information
            
        Returns:
            Dictionary with processed video data
        """
        # Try different possible path field names for compatibility
        video_path = data_info.get('file_path', data_info.get('path', ''))
        if self.data_root:
            video_path = os.path.join(self.data_root, video_path)
        
        # Try different possible text field names for compatibility
        text = data_info.get('text', data_info.get('cap', ''))
        if random.random() < self.text_drop_ratio:
            text = ''
        
        try:
            with VideoReader_contextmanager(video_path) as video_reader:
                video_length = len(video_reader)
                
                # Get middle frame
                middle_frame_idx = video_length // 2
                middle_frame = video_reader[middle_frame_idx].asnumpy()
                
                # Convert BGR to RGB if needed
                if middle_frame.shape[-1] == 3:
                    middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                middle_frame = resize_frame(middle_frame, min(self.video_sample_size))
                
                # Create static video by repeating the middle frame
                static_frames = []
                for _ in range(self.video_sample_n_frames):
                    static_frames.append(middle_frame)
                
                # Convert to numpy array
                pixel_values = np.array(static_frames)  # (T, H, W, C)
                
                # Convert to tensor format for compatibility
                # Apply transforms if bucket is disabled
                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.0
                    
                    # Apply transforms
                    pil_frames = [transforms.ToPILImage()(frame) for frame in pixel_values]
                    transformed_frames = [self.video_transforms(frame) for frame in pil_frames]
                    pixel_values = torch.stack([transforms.ToTensor()(frame) for frame in transformed_frames])
                    # Normalize to [-1, 1] range as expected
                    pixel_values = pixel_values * 2.0 - 1.0
                else:
                    # Keep as numpy array for bucket sampling
                    pass
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Fallback: create a black static video
            h, w = self.video_sample_size
            if not self.enable_bucket:
                pixel_values = torch.zeros((self.video_sample_n_frames, 3, h, w), dtype=torch.float32) - 1.0
            else:
                pixel_values = np.zeros((self.video_sample_n_frames, h, w, 3), dtype=np.uint8)
        
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "data_type": 'video',
            "path": data_info.get('path', ''),
        }
        
        return sample
    
    def _process_image(self, data_info):
        """
        Process image file to create static video by repeating the image
        
        Args:
            data_info: Dictionary containing image information
            
        Returns:
            Dictionary with processed image data as static video
        """
        # Try different possible path field names for compatibility  
        image_path = data_info.get('file_path', data_info.get('path', ''))
        if self.data_root:
            image_path = os.path.join(self.data_root, image_path)
        
        # Try different possible text field names for compatibility
        text = data_info.get('text', data_info.get('cap', ''))
        if random.random() < self.text_drop_ratio:
            text = ''
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if not self.enable_bucket:
                # Apply transforms and convert to tensor
                image = self.image_transforms(image)
                image_tensor = transforms.ToTensor()(image)
                # Normalize to [-1, 1] range as expected
                image_tensor = image_tensor * 2.0 - 1.0
                
                # Create static video by repeating the image tensor
                pixel_values = image_tensor.unsqueeze(0).repeat(self.video_sample_n_frames, 1, 1, 1)
            else:
                # Keep as numpy array for bucket sampling
                image_array = np.array(image)
                static_frames = []
                for _ in range(self.video_sample_n_frames):
                    static_frames.append(image_array)
                pixel_values = np.array(static_frames)  # (T, H, W, C)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Fallback: create a black static video
            h, w = self.image_sample_size
            if not self.enable_bucket:
                pixel_values = torch.zeros((self.video_sample_n_frames, 3, h, w), dtype=torch.float32) - 1.0
            else:
                pixel_values = np.zeros((self.video_sample_n_frames, h, w, 3), dtype=np.uint8)
        
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "data_type": 'image',
            "path": data_info.get('path', ''),
        }
        
        return sample

    def __len__(self):
        """Return the length of the dataset"""
        return self.length

    def __getitem__(self, idx):
        """Get item by index - compatible with original ImageVideoDataset interface"""
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        # Add inpaint related data if needed
        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample 