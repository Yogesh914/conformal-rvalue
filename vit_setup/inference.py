import torch
import torch.nn.functional as F
from model import create_model
from config import Config
from transformers import ViTImageProcessor
from datasets import load_from_disk
import pickle

def load_shark_image():
    """Load a great white shark image (class 2) from the validation dataset"""
    Config.apply_hf_datasets_cache()
    dataset_dir = Config.require_imagenet_dataset_dir()
    dataset = load_from_disk(str(dataset_dir))['validation']
    
    # Find indices of great white shark images (class 2)
    # shark_indices = [i for i, item in enumerate(dataset) if item['label'] == 2]
    
    # Get the first shark image
    shark_image = dataset[26587]['image']
    shark_image = shark_image.convert('RGB')
        
    # Process the image
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    processed = processor(
        images=shark_image, 
        return_tensors="pt"
    )
    return processed['pixel_values'].to(torch.float16).to(Config.DEVICE)

def load_adapter(model, adapter_path):
    """Load the saved adapter weights into the model"""
    checkpoint = torch.load(adapter_path, map_location="cpu")
    model.classifier.load_state_dict(checkpoint['adaptor'])
    return model

def get_top_predictions(logits, k=500):
    """Get top k predictions and their probabilities"""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    return top_probs.cpu().numpy(), top_indices.cpu().numpy()

def main():
    # Output path for predictions
    output_path = Config.INFERENCE_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create model and move to device
    model = create_model()
    model.eval()
    
    # Load shark image from validation dataset
    print("Loading shark image from validation dataset...")
    image_tensor = load_shark_image()
    
    # Store predictions from all adaptors
    all_predictions = {}
    
    # Find all adaptor files
    adaptor_files = sorted(Config.ADAPTER_SAVE_DIR.glob("Adaptor_*.pt"))
    print(f"Found {len(adaptor_files)} adaptor files")
    
    # Perform inference for each adaptor file
    print("Running inference...")
    for adaptor_file in adaptor_files:
        adaptor_id = adaptor_file.stem
        print(f"Processing {adaptor_id}...")
        
        # Load adapter weights
        model = load_adapter(model, adaptor_file)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs.logits.squeeze(0)  # Remove batch dimension
            
            # For this adaptor, get predictions for all NUM_MODELS outputs
            for model_idx in range(logits.shape[0]):
                # Get top 500 predictions for each model
                top_probs, top_indices = get_top_predictions(logits[model_idx], k=500)
                
                all_predictions[f'{adaptor_id}_model_{model_idx}'] = {
                    'probabilities': top_probs,
                    'indices': top_indices
                }
    
    # Save predictions to pickle file
    print(f"Saving predictions to {output_path}...")
    with output_path.open('wb') as f:
        pickle.dump(all_predictions, f)
    
    print("Done!")
    print(f"Processed predictions for {len(all_predictions)} total models")

if __name__ == "__main__":
    main()
    


