import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, AutoImageProcessor
from datasets import load_from_disk
from config import Config

class BaselineImageNet:
    def __init__(self):
        """Initialize dataset using HuggingFace's validation set"""
        Config.apply_hf_datasets_cache()
        dataset_dir = Config.require_imagenet_dataset_dir()
        self.dataset = load_from_disk(str(dataset_dir))['validation']
        # self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-384')
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        item = self.dataset[idx]
        image = item['image']  # PIL Image
        image = image.convert('RGB')
        label = item['label']
        
        processed = self.processor(
            images=image, 
            return_tensors="pt",
            do_resize=True,
            do_normalize=True
        )
        processed_image = processed['pixel_values'].squeeze(0)
        processed_image = processed_image#.to(torch.float16)
        
        return processed_image, label
    
    def __len__(self):
        """Get total number of samples"""
        return len(self.dataset)

def create_base_model():
    """Create and return the base ViT model without any adaptors"""
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')
    # model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
    # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    return model.to(Config.DEVICE)

def get_predictions(model, dataloader):
    """Run inference and return probability distributions"""
    all_probs = []
    all_labels = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running inference"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_logits), np.concatenate(all_labels)

def main():
    # Create model
    model = create_base_model()
    model.eval()
    
    # Create dataset
    full_dataset = BaselineImageNet()
    
    # Split into validation (40k) and test (10k) without shuffling
    val_indices = range(40000)  # First 40k samples
    test_indices = range(40000, 50000)  # Last 10k samples
    
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    
    # Get predictions
    print("Processing validation set (40k samples)...")
    val_probs, val_logits, val_labels = get_predictions(model, val_loader)
    
    print("Processing test set (10k samples)...")
    test_probs, test_logits, test_labels = get_predictions(model, test_loader)
    
    # # Save results
    # save_dict = {
    #     'val_probs': val_probs,
    #     'val_labels': val_labels,
    #     'test_probs': test_probs,
    #     'test_labels': test_labels,
    #     'val_logits': val_logits,
    #     'test_logits': test_logits
    # }
    
    # save_path = 'baseline_predictions_resnet18.npz'
    # np.savez(save_path, **save_dict)
    # print(f"Saved predictions to {save_path}")
    
    # Print some statistics
    val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
    test_acc = (test_probs.argmax(axis=1) == test_labels).mean()
    
    print(f"Validation accuracy (40k): {val_acc:.4f}")
    print(f"Test accuracy (10k): {test_acc:.4f}")

if __name__ == "__main__":
    main()
