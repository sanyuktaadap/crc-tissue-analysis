import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

class CRCTissueClassifier(nn.Module):
    def __init__(self, n_classes=8,  dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = dropout

        # Load AlexNet from Torch
        resnet = hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc_block = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=128, out_features=n_classes, bias=True)

    def forward(self, x, is_train=True):
        assert x.shape[1:] == (3, 227, 227), \
            f"Input shape {x.shape[1:]} not as desired!"

        out = self.resnet(x)
        out = out.view(-1, 512 * 1 * 1)
        out = self.fc_block(out)
        out = self.classifier(out)

        return out