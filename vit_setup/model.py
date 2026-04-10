import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoModelForImageClassification, ResNetForImageClassification
from config import Config

class MultiLinear(nn.Module):
    def __init__(self, num_models, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(0.02 * torch.randn(num_models, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(num_models, out_dim)) if bias else None
            
    def forward(self, x):
        if len(x.shape) == 2:
            y = torch.einsum("noi,bi->bno", self.weight, x)
        elif len(x.shape) == 3:
            assert x.shape[1] == self.weight.shape[0]
            y = torch.einsum("noi,bni->bno", self.weight, x)
        else:
            raise Exception("Invalid input shape")
            
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y

def create_model():
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')
    # model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    
    for param in model.parameters():
        param.requires_grad = False
    
    if isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[1], nn.Linear):
        num_features = model.classifier[1].in_features
    else:
        num_features = model.classifier.in_features
    
    
    adaptor = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1), #resnet
        MultiLinear(Config.NUM_MODELS, num_features, 512),
        nn.ReLU(),
        MultiLinear(Config.NUM_MODELS, 512, Config.NUM_CLASSES),
        # nn.ReLU(),
        # MultiLinear(Config.NUM_MODELS, 512, Config.NUM_CLASSES),
    )
    
    model.classifier = adaptor
    model = model.to(Config.DEVICE)
    
    return model
