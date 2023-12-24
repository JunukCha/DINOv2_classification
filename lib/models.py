import torch
import torch.nn as nn
from torchvision.models import resnet50

def load_dinov2():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14.eval()
    dinov2_vits14 = dinov2_vits14.cuda()
    return dinov2_vits14

def load_resnet50():
    resnet = resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()
    resnet = resnet.cuda()
    return resnet

def load_backbone(model="dinov2"):
    if model == "dinov2":
        backbone = load_dinov2()
    elif model == "resnet50":
        backbone = load_resnet50()
    return backbone

def load_classifier(fc_dim, checkpoint=None):
    classifier = Classifier(fc_dim)
    if checkpoint is not None:
        classifier.load_state_dict(torch.load(checkpoint))
    classifier = classifier.cuda()    
    return classifier


class Classifier(nn.Module):
    def __init__(self, fc_dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(fc_dim, 128), 
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 17), 
            nn.Softmax(), 
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x