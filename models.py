import torch
import torch.nn as nn
import torchvision.models

def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.BatchNorm2d(out_channels, eps=1e-5),
    nn.ReLU(inplace=True),
  )

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )
    
class ResNetUNet(nn.Module):
    def __init__(self, n_class, freeze=False, pretrained=False):
        super().__init__()
        self.freeze = freeze

        base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convbnrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convbnrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convbnrelu(512, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up31 = convbnrelu(256 + 512, 512, 3, 1)
        self.conv_up32 = convbnrelu(512, 512, 3, 1)
        
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up21 = convbnrelu(128 + 512, 256, 3, 1)
        self.conv_up22 = convbnrelu(256, 256, 3, 1)
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up11 = convbnrelu(64 + 256, 128, 3, 1)
        self.conv_up12 = convbnrelu(128, 128, 3, 1)
        
        self.upsample0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up01 = convbnrelu(64 + 128, 64, 3, 1)
        self.conv_up02 = convbnrelu(64, 64, 3, 1)
        
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_original_size = convbnrelu(64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

#         layer4 = self.layer4_1x1(layer4)
        x = self.upsample3(layer4)
#         layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
#         layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
#         layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
#         layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)

        x = self.upsample(x)
        x = self.conv_original_size(x)

        out = self.conv_last(x)

        return out, x
    
class ResNetBackBone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        
        self.avgpool = self.base_layers[8]
        
        # projection head
        self.fc1 = nn.Linear(512, 128, bias=False)
#         self.fc2 = nn.Linear(256, 64, 1)

    def forward(self, input):
        

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        x = self.avgpool(layer4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
    
class ResNetUNetHead(nn.Module):
    def __init__(self, freeze=True, pretrained=False):
        super().__init__()
        self.freeze = freeze

        base_model = torchvision.models.resnet18(pretrained=pretrained)
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convbnrelu(128, 128, 1, 0)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convbnrelu(256, 256, 1, 0)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convbnrelu(512, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up31 = convbnrelu(256 + 512, 512, 3, 1)
        self.conv_up32 = convbnrelu(512, 512, 3, 1)
        
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up21 = convbnrelu(128 + 512, 256, 3, 1)
        self.conv_up22 = convbnrelu(256, 256, 3, 1)
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up11 = convbnrelu(64 + 256, 128, 3, 1)
        self.conv_up12 = convbnrelu(128, 128, 3, 1)
        
        self.upsample0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up01 = convbnrelu(64 + 128, 64, 3, 1)
        self.conv_up02 = convbnrelu(64, 64, 3, 1)
        
        # projection head
        self.conv_proh1 = nn.Conv2d(64, 64, 1)
        self.conv_proh2 = nn.Conv2d(64, 32, 1)

    def forward(self, input):
        
        if self.freeze:
            with torch.no_grad():
                layer0 = self.layer0(input)
                layer1 = self.layer1(layer0)
                layer2 = self.layer2(layer1)
                layer3 = self.layer3(layer2)
                layer4 = self.layer4(layer3)
        else:
            layer0 = self.layer0(input)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

        x = self.upsample3(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)
        
        x = self.conv_proh1(x)
        x = self.conv_proh2(x)

        return x
    
class ResNetUNetHeadOneStage(nn.Module):
    def __init__(self, freeze=True, pretrained=False):
        super().__init__()
        self.freeze = freeze

        base_model = torchvision.models.resnet18(pretrained=pretrained)
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convbnrelu(128, 128, 1, 0)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convbnrelu(256, 256, 1, 0)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convbnrelu(512, 512, 1, 0)
        
        self.avgpool = base_layers[8]
        
        # projection head
        self.fc1 = nn.Linear(512, 128, bias=False)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up31 = convbnrelu(256 + 512, 512, 3, 1)
        self.conv_up32 = convbnrelu(512, 512, 3, 1)
        
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up21 = convbnrelu(128 + 512, 256, 3, 1)
        self.conv_up22 = convbnrelu(256, 256, 3, 1)
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up11 = convbnrelu(64 + 256, 128, 3, 1)
        self.conv_up12 = convbnrelu(128, 128, 3, 1)
        
        self.upsample0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up01 = convbnrelu(64 + 128, 64, 3, 1)
        self.conv_up02 = convbnrelu(64, 64, 3, 1)
        
        # projection head
        self.conv_proh1 = nn.Conv2d(64, 64, 1)
        self.conv_proh2 = nn.Conv2d(64, 32, 1)

    def forward(self, input):
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        x_e = self.avgpool(layer4)
        x_e = torch.flatten(x_e, 1)
        x_e = self.fc1(x_e)

        x = self.upsample3(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)
        
        x = self.conv_proh1(x)
        x = self.conv_proh2(x)

        return x, x_e