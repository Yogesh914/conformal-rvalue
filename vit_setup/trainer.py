import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from config import Config

class Trainer:
    def __init__(self, model, train_loader, optimizer=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer or torch.optim.Adam(
            model.classifier.parameters(), 
            lr=Config.LEARNING_RATE
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.scaler = GradScaler("cuda", enabled=Config.DEVICE.type == "cuda")
        
    def train(self):
        start_time = time.time()
        
        for ep in range(1, Config.EPOCHS + 1):
            
            train_bar = tqdm(self.train_loader, 
                           desc=f'Epoch {ep}/{Config.EPOCHS} - Training',
                           leave=False)
            
            for ims, labs, w in train_bar:
            
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(Config.DEVICE.type, enabled=Config.DEVICE.type == "cuda"):
                    pred = self.model(ims.to(Config.DEVICE)).logits
                    pred_reshape = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
                    labels = labs.to(Config.DEVICE)[:,None].repeat(1,pred.shape[1])
                    labels_reshape = labels.reshape(pred.shape[0] * pred.shape[1])
                    
                    losses = self.loss_fn(pred_reshape, labels_reshape).reshape(pred.shape[0], pred.shape[1])
                    weighted_loss = (losses * w.to(Config.DEVICE))
                    loss = torch.mean(weighted_loss)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                del pred, pred_reshape, labels, labels_reshape, losses, weighted_loss, loss
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
    
    def save_model(self, arg_index):
        try:
            model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            Config.ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            
            adapter_save_path = Config.ADAPTER_SAVE_DIR / f"Adaptor_{arg_index}.pt"
            
            save_object = {
                'weight': self.train_loader.dataset.weights,
                'adaptor': model_module.classifier.state_dict()
            }
            
            torch.save(save_object, adapter_save_path)
            print(f"Successfully saved adapter to {adapter_save_path}")
            
        except Exception as e:
            print(f"Error saving adapter: {str(e)}")
            raise
