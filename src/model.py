import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1

class FaceNetModel(nn.Module):
    def __init__(self, classify=False, num_classes=None):
        super().__init__()
        self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        self.classify = classify
        if classify:
            assert num_classes is not None
            self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        emb = self.encoder(x)
        if self.classify:
            return self.head(emb)
        return emb
