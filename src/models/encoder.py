from setup.Modules import *

class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.ResNet18 = resnet.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT, progress=True)
        self.ResNet18.fc = nn.Linear(self.ResNet18.fc.in_features, embedding_size)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.ResNet18(images)
        return self.Dropout(self.ReLU(features))
