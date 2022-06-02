import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self,indim,middim,outdim,stride=1):
        super(ResidualBlock, self).__init__()
        self.Bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=indim, out_channels=middim, kernel_size=1,stride=1),
            nn.BatchNorm2d(num_features=middim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=middim, out_channels=middim, kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(num_features=middim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=middim, out_channels=outdim, kernel_size=1),
            nn.BatchNorm2d(num_features=outdim),
            nn.ReLU(inplace=True)
        )
        self.iden = nn.Sequential(
            nn.Conv2d(in_channels=indim, out_channels=outdim,kernel_size=1, stride=stride),
            nn.BatchNorm2d(outdim)
        )
    def forward(self,x):
        output = self.iden(x) + self.Bottleneck(x)
        return output
class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64,64,256),
            ResidualBlock(256,64,256,stride=2),
            ResidualBlock(256,64,256)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(256,128,512,stride=2),
            ResidualBlock(512,128,512),
            ResidualBlock(512,128,512),
            ResidualBlock(512,128,512)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(512,256,1024,stride=2),
            ResidualBlock(1024,256,1024),
            ResidualBlock(1024,256,1024,stride=2),
            ResidualBlock(1024,256,1024),
            ResidualBlock(1024,256,1024,stride=2),
            ResidualBlock(1024,256,1024)
        )

        self.layer5 = nn.Sequential(
            ResidualBlock(1024,512,2048,stride=2),
            ResidualBlock(2048,1024,2048),
            ResidualBlock(2048,1024,2048)
        )

        self.fc = nn.Linear(2048, 5)
        self.avgpool = nn.AvgPool2d((2,2))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
