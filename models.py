from torch import nn
import torch
import torch.nn.functional as F
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
        
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
    
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
    
class RDCAB(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(RDCAB, self).__init__()
        #Split Mechanism
        distillation_rate=0.25
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        gc = growth_rate
        fc = 48
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels + 0 * gc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels + 1 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels + 2 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels + 3 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels + 4 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels + 5 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Conv2d(in_channels + 6 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(nn.Conv2d(in_channels + 7 * fc, gc, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        #Local Feature Fusion
        self.lff = nn.Conv2d(128, 64, kernel_size=1)
        #Contrast Channle Attention 
        self.contrast = stdv_channels
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 16, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        

    def forward(self, x): 
        Local_Residual = x
        
        layer1 = self.layer1(x) #64->64
        distilled_c1, remaining_c1 = torch.split(layer1, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer2 = self.layer2(torch.cat((x, remaining_c1), 1)) # 112->64
        distilled_c2, remaining_c2 = torch.split(layer2, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer3 = self.layer3(torch.cat((x, remaining_c1, remaining_c2), 1)) # 160->48
        distilled_c3, remaining_c3 = torch.split(layer3, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer4 = self.layer4(torch.cat((x, remaining_c1, remaining_c2, remaining_c3), 1)) #208->64
        distilled_c4, remaining_c4 = torch.split(layer4, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer5 = self.layer5(torch.cat((x, remaining_c1, remaining_c2, remaining_c3,remaining_c4), 1)) #256->64
        distilled_c5, remaining_c5 = torch.split(layer5, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer6 = self.layer6(torch.cat((x, remaining_c1, remaining_c2, remaining_c3,remaining_c4,remaining_c5), 1)) #304->64
        distilled_c6, remaining_c6 = torch.split(layer6, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer7 = self.layer7(torch.cat((x, remaining_c1, remaining_c2, remaining_c3,remaining_c4,remaining_c5,remaining_c6), 1)) #352->64
        distilled_c7, remaining_c7 = torch.split(layer7, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        layer8 = self.layer8(torch.cat((x, remaining_c1, remaining_c2, remaining_c3,remaining_c4,remaining_c5,remaining_c6,remaining_c7), 1)) #400->64
        distilled_c8, remaining_c8 = torch.split(layer8, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        
        out = torch.cat([distilled_c1,distilled_c2,distilled_c3,distilled_c4,distilled_c5,distilled_c6,distilled_c7,distilled_c8], dim=1) 
        x = self.lff(out)
        
        y =self.contrast(x)+self.avg_pool(x)
        y = self.conv_du(y)
        x = x*y
        x = x+Local_Residual
        return x


class RecursiveBlock(nn.Module):
    def __init__(self,num_channels, num_features, growth_rate, B, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.G0 = num_features
        self.G = growth_rate
        self.rdbs = RDCAB(self.G0, self.G) #residual dense channel attention block & Split Mechanism
        
    def forward(self, sfe2):
        global concat_LF
        x=sfe2
        local_features = []
        for i in range(self.U):
            x = self.rdbs(x)
            local_features.append(x)    
        concat_LF = torch.cat(local_features, 1)
        return x
        
class DRRDB(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, B, U):
        super(DRRDB, self).__init__()
        self.B = B
        self.G0 = num_features
        self.G = growth_rate
        self.U = U
        self.scale_factor=scale_factor
        self.num_channels=num_channels
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        
        self.recursive_SR = nn.Sequential(*[RecursiveBlock(num_channels if i==0 else num_features,
        num_features,
        growth_rate, 
        B, 
        U) for i in range(B)])
        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.U * self.B, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        # Upscale & Reconstruction
        self.upscale1 = nn.Conv2d(self.G0,48,self.num_channels,padding=3//2,dilation=1)
        self.upscale1_scale=Scale(1)
        self.pixelshuffle=nn.PixelShuffle(self.scale_factor)
    def forward(self, x):
        x_up = F.interpolate(x, mode='bicubic',scale_factor=self.scale_factor)
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        local_global_features=[]
        for i in range(self.B):
            if i==0:
                x= self.recursive_SR(sfe2)
                local_global_features.append(concat_LF)

            elif i>0:
                x= self.recursive_SR(x)
                local_global_features.append(concat_LF)

        x = self.gff(torch.cat(local_global_features, 1)) + sfe1
        x = self.pixelshuffle(self.upscale1_scale(self.upscale1(x)))+x_up
        return x
