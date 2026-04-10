import os
import sys
from config import Config
from dataset import ImageNet, WeightedImageNet
from model import create_model
from trainer import Trainer
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


def main():
            
    save_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    # Create datasets
    train_data = ImageNet(split='train')
    weighted_train_data = WeightedImageNet(train_data)
    
    # Create data loader
    train_loader = DataLoader(
        weighted_train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    
    # Create and train model
    model = create_model()
    trainer = Trainer(model, train_loader)
    trainer.train()
    trainer.save_model(save_id)

if __name__ == "__main__":
    main()