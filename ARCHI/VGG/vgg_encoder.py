import torch
import torch.nn as nn

import torch.nn.functional as F


def get_configs(arch='vgg16'):

    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")
    
    return configs



class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False, high_freq_residual=True, pyramid=True, skips=3, pyramid_version="v1"):

        super(VGGEncoder, self).__init__()

        self.pyramid = pyramid
        self.skips = skips
        self.high_freq_residual = high_freq_residual
        self.pyramid_version = pyramid_version
        
        if self.high_freq_residual:
            print("activate high-frequency redisual connection")
        if self.pyramid:
            print("use pyramid high-frequency redisual connection")

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder1 = EncoderBlock(input_dim=3,   output_dim=64,  hidden_dim=64,  layers=configs[0], enable_bn=enable_bn)
        self.encoder2 = EncoderBlock(input_dim=64,  output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.encoder3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.encoder4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)

        if (self.high_freq_residual==True) and (self.pyramid==True):
            if self.pyramid_version == "v1":
                print("=> pyramid version: v1")
                if self.skips == 3:
                    self.pyramid3 = FPN_photowct2(level=3)
                    self.pyramid2 = FPN_photowct2(level=2)
                elif self.skips == 4:
                    self.pyramid3 = FPN_photowct2(level=3)
                    self.pyramid2 = FPN_photowct2(level=2)
                    self.pyramid1 = FPN_photowct2(level=1)
            elif self.pyramid_version == "v2":
                print("=> pyramid version: v2")
                if self.skips == 3:
                    self.pyramid_model = FPNv2()
                elif self.skips == 4:
                    self.pyramid_model = FPN()


    def forward(self, x0):

        x1, skip1 = self.encoder1(x0) # 256->128
        x2, skip2 = self.encoder2(x1) # 128->64
        x3, skip3 = self.encoder3(x2) # 64->32
        x4, skip4 = self.encoder4(x3) # 32->16


        if self.high_freq_residual != True:
            return None, None, None, None, x4

        if self.skips == 3:
            skip1 = None

        # pyramid fusion v2
        if self.pyramid == True:
            if self.pyramid_version == "v1":
                if self.skips == 3:
                    skip3_enhance = self.pyramid3([skip3,skip4])
                    skip2_enhance = self.pyramid2([skip2,skip3,skip4])
                    return skip1, skip2_enhance, skip3_enhance, skip4, x4
                elif self.skips == 4:
                    skip3_enhance = self.pyramid3([skip3, skip4])
                    skip2_enhance = self.pyramid2([skip2, skip3, skip4])
                    skip1_enhance = self.pyramid1([skip1, skip2, skip3, skip4])
                    return skip1_enhance, skip2_enhance, skip3_enhance, skip4, x4
            elif self.pyramid_version == "v2":
                if self.skips == 3:
                    skip2_enhance, skip3_enhance = self.pyramid_model(skip2,skip3,skip4)
                    return skip1, skip2_enhance, skip3_enhance, skip4, x4
                elif self.skips == 4:
                    skip1_enhance, skip2_enhance, skip3_enhance = self.pyramid_model(skip1,skip2,skip3,skip4)
                    return skip1_enhance, skip2_enhance, skip3_enhance, skip4, x4
        else:
            return skip1, skip2, skip3, skip4, x4


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        self.conv = nn.Sequential()
        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.conv.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.conv.add_module('%d EncoderLayer' % i, layer)

        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.biupsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
    
    def forward(self, x):
        for name, layer in self.conv.named_children():
            # print(name)
            x = layer(x)

        skip = x
        x = self.pool(x)
        skip = skip - self.biupsample(x)
        return x,skip



def Normalize(in_channels):
    if in_channels<32:
        num_groups = 1
    else:
        num_groups = 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                # nn.GroupNorm(num_groups=32, num_channels=output_dim, eps=1e-6, affine=True),
                nn.PReLU(),
            )

        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
            )
    
    def forward(self, x):

        return self.layer(x)




class FPNv2(nn.Module):
    def __init__(self):
        super(FPNv2, self).__init__()

        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        # Lateral layers
        self.laterallayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.laterallayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        # self.laterallayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        # Final conv layers
        self.finalconv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.finalconv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                        padding=1, bias=False)
        # self.finalconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
        #                 padding=1, bias=False)

    def forward(self, skip2, skip3, skip4): # 由浅入深
        
        p4 = self.toplayer(skip4)
        p3 = self._upsample_add(p4, self.laterallayer1(skip3))
        p2 = self._upsample_add(p3, self.laterallayer2(skip2))
        # p1 = self._upsample_add(p2, self.laterallayer3(skip1))

        # Final conv layers
        p3_e = self.finalconv1(p3)
        p2_e = self.finalconv2(p2)
        # p1_e = self.finalconv3(p1)

        return p2_e, p3_e

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y





class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        # Lateral layers
        self.laterallayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.laterallayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.laterallayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        # Final conv layers
        self.finalconv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.finalconv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                        padding=1, bias=False)
        self.finalconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)

    def forward(self, skip1, skip2, skip3, skip4): # 由浅入深
        
        p4 = self.toplayer(skip4)
        p3 = self._upsample_add(p4, self.laterallayer1(skip3))
        p2 = self._upsample_add(p3, self.laterallayer2(skip2))
        p1 = self._upsample_add(p2, self.laterallayer3(skip1))

        # Final conv layers
        p3_e = self.finalconv1(p3)
        p2_e = self.finalconv2(p2)
        p1_e = self.finalconv3(p1)

        return p1_e, p2_e, p3_e

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y







class FPN_photowct2(nn.Module):
    def __init__(self,level):
        super(FPN_photowct2, self).__init__()

        self.level = level

        if level == 3:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0),
            )
        elif level == 2:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=896, out_channels=128, kernel_size=1, stride=1, padding=0),
            )
        elif level == 1:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=960, out_channels=64, kernel_size=1, stride=1, padding=0),
            )
        self.biupsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大

    def forward(self,x):
        
        if self.level == 3:
            up_x1 = self.biupsample(x[1])
            x_combined = torch.cat([x[0], up_x1], dim=1)
            skip = self.layer(x_combined)
            return skip
        elif self.level == 2:
            up_x1 = self.biupsample(x[1])
            up_x2 = self.biupsample(self.biupsample(x[2]))
            x_combined = torch.cat([x[0], up_x1, up_x2], dim=1)
            skip = self.layer(x_combined)
            return skip
        elif self.level == 1:
            up_x1 = self.biupsample(x[1])
            up_x2 = self.biupsample(self.biupsample(x[2]))
            up_x3 = self.biupsample(self.biupsample(self.biupsample(x[3])))
            x_combined = torch.cat([x[0], up_x1, up_x2, up_x3], dim=1)
            skip = self.layer(x_combined)
            return skip

if __name__ == "__main__":

    input = torch.randn((5,3,224,224))

    configs = get_configs('vgg16')
    print(configs)

    model = VGGEncoder(configs)

    output = model(input)

    print(output[0].shape)
