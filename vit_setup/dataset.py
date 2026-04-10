import torch
from torch.utils.data import Dataset
from torch.distributions.dirichlet import Dirichlet
from transformers import AutoImageProcessor
from datasets import load_from_disk
from config import Config
import warnings


class ImageNet(Dataset):
    def __init__(self, split='train'):
        Config.apply_hf_datasets_cache()
        dataset_dir = Config.require_imagenet_dataset_dir()
        self.dataset = load_from_disk(str(dataset_dir))[split]
        # self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)
        # self.processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-384', use_fast=True)
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
        item = self.dataset[idx]
        image = item['image']
        image = image.convert('RGB')
        
        label = item['label']
        
        processed = self.processor(
            images=image, 
            return_tensors="pt"
        )
        processed_image = processed['pixel_values'].squeeze(0)
        processed_image = processed_image#.to(torch.float16)
        
        return processed_image, label
    
    def __len__(self):
        return len(self.dataset)

class WeightedImageNet(Dataset):
    def __init__(self, base_dataset, num_model=Config.NUM_MODELS):
        self.base_dataset = base_dataset
        alpha = torch.ones(len(self.base_dataset))
        dirichlet = Dirichlet(alpha)
        self.weights = torch.stack([dirichlet.sample() * len(self.base_dataset) for _ in range(num_model)])
        
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        weight = self.weights[:, idx]
        return image, label, weight
    
    def __len__(self):
        return len(self.base_dataset)
