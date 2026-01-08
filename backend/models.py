import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(b)
        # Handle simple padding issues if sizes don't match exactly due to odd dims
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:])
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
             d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
