import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

class CRCTextureClassifier(nn.Module):
    def __init__(self, n_classes=8,  dropout=0.2):
        super().__init__()

        self.n_classes = n_classes
        self.dropout = dropout

        # Load AlexNet from Torch
        alexenet = hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        self.feature_extractor = nn.Sequential(
            alexenet.features,
            nn.AdaptiveAvgPool2d(output_size=(6, 6))
        )
        self.fc_block = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.LeakyReLU(inplace=True)
        )
        self.classifier = nn.Linear(in_features=1024, out_features=n_classes, bias=True)

    def forward(self, x, is_train=True):
        assert x.shape[1:] == (3, 224, 224), \
            f"Input shape {x.shape[1:]} not as desired!"

        out = self.feature_extractor(x)
        out = self.fc_block(out)
        out = F.dropout(out, p=self.dropout, training=is_train)
        out = self.classifier(out)

        return out