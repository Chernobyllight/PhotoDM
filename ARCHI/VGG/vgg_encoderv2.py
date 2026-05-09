import torch.nn.init as init
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1.0)
        init.constant_(m.bias, 0.0)


class VGGEncoderv2(nn.Module):

    def __init__(self, vgg_type="vgg19", pool_method="average",
            pretrained_vgg_resume=None, pyramid_version="v1",
            high_freq_residual=True, pyramid=True,skips=3):
        super(VGGEncoderv2, self).__init__()

        self.pool_method = pool_method
        self.pyramid = pyramid
        self.skips = skips
        self.high_freq_residual = high_freq_residual
        self.pyramid_version = pyramid_version
        
        if self.high_freq_residual:
            print("=> activate high-frequency redisual connection")
        if self.pyramid:
            print("=> use pyramid high-frequency redisual connection")

        if vgg_type=="vgg19":
            vgg19 = models.vgg19(weights=None)
            if pretrained_vgg_resume != None:
                print("=> load pretrained vgg as encoder")
                checkpoint = torch.load(pretrained_vgg_resume, weights_only=True)  # 可能是 .pth 或 .ckpt
                vgg19.load_state_dict(checkpoint)  # 如果 checkpoint 直接是 state_dict

            self.encoder1 = vgg19.features[0:4]
            self.encoder2 = vgg19.features[5:9]
            self.encoder3 = vgg19.features[10:18]
            self.encoder4 = vgg19.features[19:27]


            if pool_method == "average":
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
            elif pool_method == "max":
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        

        elif vgg_type=="vgg16":
            vgg16 = models.vgg16(weights=None)
            if pretrained_vgg_resume != None:
                print("=> load pretrained vgg as encoder")
                checkpoint = torch.load(pretrained_vgg_resume, weights_only=True)  # 可能是 .pth 或 .ckpt
                vgg16.load_state_dict(checkpoint)  # 如果 checkpoint 直接是 state_dict

            self.encoder1 = vgg16.features[0:4]
            self.encoder2 = vgg16.features[5:9]
            self.encoder3 = vgg16.features[10:16]
            self.encoder4 = vgg16.features[17:23]


            if pool_method == "average":
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
                self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍放大
            elif pool_method == "max":
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
                self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)


        self.encoder1 = self.freeze_model(self.encoder1)
        self.encoder2 = self.freeze_model(self.encoder2)
        self.encoder3 = self.freeze_model(self.encoder3)
        self.encoder4 = self.freeze_model(self.encoder4)

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
                self.pyramid_model = FPN()


    # def encoder_forward(self,encoder,input_x):
        

    def forward(self, x0):

        self.encoder1 = self.freeze_model(self.encoder1)
        self.encoder2 = self.freeze_model(self.encoder2)
        self.encoder3 = self.freeze_model(self.encoder3)
        self.encoder4 = self.freeze_model(self.encoder4)


        x1 = self.encoder1(x0) # 256->128
        skip1 = x1
        if self.pool_method == "average":
            x1 = self.pool1(x1)
            skip1 = skip1 - self.unpool1(x1)
        else:
            x1, indices = self.pool1(x1)
            skip1 = skip1 - self.unpool1(x1, indices)

        x2 = self.encoder2(x1) # 128->64
        skip2 = x2
        if self.pool_method == "average":
            x2 = self.pool2(x2)
            skip2 = skip2 - self.unpool2(x2)
        else:
            x2, indices = self.pool2(x2)
            skip2 = skip2 - self.unpool2(x2,indices)

        x3 = self.encoder3(x2) # 128->64
        skip3 = x3
        if self.pool_method == "average":
            x3 = self.pool3(x3)
            skip3 = skip3 - self.unpool3(x3)
        else:
            x3, indices = self.pool3(x3)
            skip3 = skip3 - self.unpool3(x3,indices)


        x4 = self.encoder4(x3) # 128->64
        skip4 = x4
        if self.pool_method == "average":
            x4 = self.pool4(x4)
            skip4 = skip4 - self.unpool4(x4)
        else:
            x4, indices = self.pool4(x4)
            skip4 = skip4 - self.unpool1(x4,indices)

        if  self.high_freq_residual != True:
            return None, None, None, None, x4

        if self.skips == 3:
            skip1 = None
        # pyramid fusion v2
        if self.pyramid==True:
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
                skip1_enhance, skip2_enhance, skip3_enhance = self.pyramid_model(skip1,skip2,skip3,skip4)
                return skip1_enhance, skip2_enhance, skip3_enhance, skip4, x4
        else:
            return skip1, skip2, skip3, skip4, x4


    def freeze_model(self, model):
        for (name, param) in model.named_parameters():
            param.requires_grad = False
        
        return model



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
        # /group/40063/chernoliu/style_4layer_kv_fapnv5/vgg_ckpt/vgg16-397923af.pth
        # /group/40063/chernoliu/style_4layer_kv_fapnv5/vgg_ckpt/vgg19-dcbb9e9d.pth
    input = torch.randn((2,3,224,224))

    model = VGGEncoderv2(vgg_type="vgg16", pool_method="average",pyramid=True, 
            pretrained_vgg_resume="/group/40063/chernoliu/style_4layer_kv_fapnv5/vgg_ckpt/vgg16-397923af.pth")

    output = model(input)

    print(output[0].shape)
