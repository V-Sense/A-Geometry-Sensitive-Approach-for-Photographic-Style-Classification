import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
import torch
        

def denorm(tensor, mean, std):
    pdb.set_trace()
    for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)            
    return tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class ResNet_Direct_Saliency(nn.Module):
    def __init__(self, dr = 0.0, pretrained = True, freeze = False, pooling  = True):
        super(ResNet_Direct_Saliency, self).__init__()
        self.net = models.resnet152(pretrained)
        self.pooling = pooling
        self.downsampler = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3)
        self.downsampler.apply(weights_init)
        self.fc = nn.Linear(self.net.fc.in_features + 3136 ,14)
        self.net.fc = None
    def forward(self, features):
        #pdb.set_trace()    
        chunks = features.split([3, 1],1);
        #chunk out saliency
        x = chunks[0];
        z = chunks[1];        
        
        #convolve RGB        
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)

        if self.pooling:
            x = self.net.maxpool(x)
        else:
            x = self.downsampler(x)
            
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        
        #featuresize 224*224
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)
        #featuresize 56*56
        
        z = z.view(z.size(0), -1)
        
        x = torch.cat((x,z),1)
        y = self.fc(x)
        
        return y
        
class DenseNet_Direct_Saliency(nn.Module):
    def __init__(self, pretrained, pooling):
        super(DenseNet_Direct_Saliency, self).__init__()
        self.net = models.densenet161(pretrained)
        self.fc = nn.Linear(self.net.classifier.in_features + 3136 ,14)
        self.net.classifier = None
        
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.conv_1.apply(weights_init)
        self.bn1 = nn.BatchNorm2d(1)
        
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.conv_2.apply(weights_init)
        self.bn2 = nn.BatchNorm2d(1)
        
        self.pooling  = pooling
            
    def forward(self, features):
        #pdb.set_trace()    
        chunks = features.split([3, 1],1);
        #chunk out saliency
        x = chunks[0];
        z = chunks[1];        
        
        x = self.net.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(features.size(0), -1)
        
        if self.pooling:
            z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)
            z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)
            
        else: # Do strided convolutions  if pooling is disabled
            z = self.conv_1(z)
            z = self.bn1(z)
            z = F.leaky_relu(z)
            
            z = self.conv_2(z)
            z = self.bn2(z)
            z = F.leaky_relu(z)            
        
        z = z.view(z.size(0), -1)
        x = torch.cat((x,z),1)
        y = self.fc(x)
        
        return y                

    
def resnet_saliency_direct(pretrained, pooling):
    model = ResNet_Direct_Saliency(pretrained, pooling)
    return model        
    
def densenet_saliency_direct(pretrained, pooling):
    model = DenseNet_Direct_Saliency(pretrained, pooling)
    return model        