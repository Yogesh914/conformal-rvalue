import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from model import create_model
from config import Config
from transformers import AutoImageProcessor
from datasets import load_from_disk
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class AdapterImageNet:
    def __init__(self):
        Config.apply_hf_datasets_cache()
        dataset_dir = Config.require_imagenet_dataset_dir()
        self.dataset = load_from_disk(str(dataset_dir))['validation']
        # self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-384')
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        image = image.convert('RGB')
        label = item['label']
        
        processed = self.processor(
            images=image, 
            return_tensors="pt",
            do_resize=True,
            do_normalize=True
        )
        processed_image = processed['pixel_values'].squeeze(0)
        processed_image = processed_image
        
        return processed_image, label
    
    def __len__(self):
        return len(self.dataset)

def load_adapter(model, adapter_path):
    checkpoint = torch.load(adapter_path, map_location="cpu")
    model.classifier.load_state_dict(checkpoint['adaptor'])
    return model

def get_val_predictions(model, dataloader):
    """Get predictions and true label logits for validation set"""
    all_preds = []
    all_true_logits = []
    all_labels = []
    all_true_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running validation inference"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            logits = outputs.logits.to(dtype=torch.float16)  # [batch_size, num_models, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [batch_size, num_models, num_classes]
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)  # [batch_size, num_models]
            
            # Get true label logits
            true_logits = logits[torch.arange(len(labels))[:, None], 
                               torch.arange(logits.shape[1])[None, :],
                               labels[:, None]].to(dtype=torch.float16)  # [batch_size, num_models]

            true_probs = probs[torch.arange(len(labels))[:, None],
                                 torch.arange(probs.shape[1])[None, :],
                                 labels[:, None]].to(dtype=torch.float16)  # [batch_size, num_models]
            
            all_preds.append(preds.cpu().numpy())
            all_true_logits.append(true_logits.cpu().numpy())
            all_true_probs.append(true_probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return (np.concatenate(all_preds), 
            np.concatenate(all_true_logits),
            np.concatenate(all_true_probs),
            np.concatenate(all_labels))

def get_test_predictions(model, dataloader):
    """Get full logits distributions for test set"""
    all_preds = []
    all_logits = []
    all_labels = []
    all_probs = []
    all_true_logits = []
    all_true_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running test inference"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            logits = outputs.logits.to(dtype=torch.float16)  # [batch_size, num_models, num_classes]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            true_logits = logits[torch.arange(len(labels))[:, None],
                                 torch.arange(logits.shape[1])[None, :],
                                 labels[:, None]].to(dtype=torch.float16)  # [batch_size, num_models]
            
            true_probs = probs[torch.arange(len(labels))[:, None],
                                 torch.arange(probs.shape[1])[None, :],
                                 labels[:, None]].to(dtype=torch.float16)
            
            
            all_preds.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_true_logits.append(true_logits.cpu().numpy())
            all_true_probs.append(true_probs.cpu().numpy())
    
    return (np.concatenate(all_preds),
            np.concatenate(all_logits),
            np.concatenate(all_probs),
            np.concatenate(all_true_logits),
            np.concatenate(all_true_probs),
            np.concatenate(all_labels))

def main():
    model = create_model()
    model.eval()
    
    # Create dataset
    full_dataset = AdapterImageNet()
    
    # Split into validation (40k) and test (10k) without shuffling
    val_indices = range(40000)
    test_indices = range(40000, 50000)
    
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Find all adaptor files
    adaptor_files = sorted(Config.ADAPTER_SAVE_DIR.glob("Adaptor_*.pt"))
    print(f"Found {len(adaptor_files)} adaptor files")
    
    # Lists to store all results
    all_val_true_logits = []  # Will store only true label logits for validation
    all_val_true_probs = []
    all_test_logits = []  # Will store full logits distribution for test
    all_test_true_logits = []
    all_test_true_probs = []
    all_test_probs = []
    all_val_accs = []
    all_test_accs = []
    all_val_labels = []
    all_test_labels = []
    
    
    # Evaluate each adapter
    for adaptor_file in adaptor_files:
        adaptor_id = adaptor_file.stem
        print(f"\nProcessing {adaptor_id}...")
        
        # Load adapter weights
        model = load_adapter(model, adaptor_file)
        
        # Get validation predictions (true label logits only)
        print("Processing validation set (40k samples)...")
        val_preds, val_true_logits, val_true_probs, val_labels = get_val_predictions(model, val_loader)
        
        # Get test predictions (full logits distribution)
        print("Processing test set (10k samples)...")
        test_preds, test_full_logits, test_full_probs, test_true_logits, test_true_probs, test_labels = get_test_predictions(model, test_loader)
        
        # Calculate accuracies
        val_accs = (val_preds == val_labels[:, None]).mean(axis=0)
        test_accs = (test_preds == test_labels[:, None]).mean(axis=0)
        
        # Store results
        all_val_true_logits.append(val_true_logits)
        all_val_true_probs.append(val_true_probs)
        all_test_logits.append(test_full_logits)
        all_test_true_logits.append(test_true_logits)
        all_test_true_probs.append(test_true_probs)
        all_test_probs.append(test_full_probs)
        all_val_accs.append(val_accs)
        all_test_accs.append(test_accs)
        # all_val_labels.append(val_labels)
        # all_test_labels.append(test_labels)
    
    # # Stack validation results
    all_val_true_logits = np.concatenate(all_val_true_logits, axis=1)  # [40000, total_models]
    all_val_true_probs = np.concatenate(all_val_true_probs, axis=1)  # [40000, total_models]
    
    # # Stack test results
    all_test_logits = np.concatenate(all_test_logits, axis=1)  # [10000, total_models, 1000]
    all_test_true_logits = np.concatenate(all_test_true_logits, axis=1)  # [10000, total_models]
    all_test_true_probs = np.concatenate(all_test_true_probs, axis=1)  # [10000, total_models]
    all_test_probs = np.concatenate(all_test_probs, axis=1)  # [10000, total_models, 1000]
    
    # Stack accuracies
    all_val_accs = np.concatenate(all_val_accs)
    all_test_accs = np.concatenate(all_test_accs)
    
    # Stack labels
    # all_val_labels = np.concatenate(all_val_labels)
    # all_test_labels = np.concatenate(all_test_labels)
    
    #print accuracy across all models and variance
    print(f"Validation accuracy (40k): {all_val_accs.mean():.4f} ± {all_val_accs.std():.4f}")
    print(f"Test accuracy (10k): {all_test_accs.mean():.4f} ± {all_test_accs.std():.4f}")
    
    # Save accuracies
    # save_path = './accuracy_info_vit_l.npz'
    # np.savez(save_path, val_accs=all_val_accs, test_accs=all_test_accs)
    
    # label_save_dict = {
    #     'val_labels': all_val_labels,
    #     'test_labels': all_test_labels
    # }

    # label_save_path = "./data/labels_info.npz"
    # np.savez(label_save_path, **label_save_dict)
    
    save_dir = Config.EVAL_OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # # Save each array individually as .npy
    np.save(save_dir / 'val_true_logits.npy', all_val_true_logits)
    np.save(save_dir / 'val_true_probs.npy', all_val_true_probs)
    np.save(save_dir / 'test_logits.npy', all_test_logits)
    np.save(save_dir / 'test_true_logits.npy', all_test_true_logits)
    np.save(save_dir / 'test_true_probs.npy', all_test_true_probs)
    np.save(save_dir / 'test_probs.npy', all_test_probs)
    
    
if __name__ == "__main__":
    main()
