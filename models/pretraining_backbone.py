import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.resnet import resnet18


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = IntermediateLayerGetter(resnet18(pretrained=pretrained), {"avgpool": "out"}).to(device)
        self.fc = nn.Linear(512, 4, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.features(x)["out"]
        x = torch.flatten(x, 1)
        return self.fc(x)
