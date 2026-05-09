import torch
import torch.nn as nn
from ARCHI.TRANSFORM.attn_trans import StyleTransferBlock_vggv1, StyleTransferBlock_vggv2

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



class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False,
                decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=True
                    ):

        super(VGGDecoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stages in VGG")


        self.decoder_attn = decoder_attn
        if decoder_attn=="v1":
            print("=> build attention module: attn v1")
            self.attn4 = StyleTransferBlock_vggv1(channels=512, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn3 = StyleTransferBlock_vggv1(channels=256, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn2 = StyleTransferBlock_vggv1(channels=128, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn1 = StyleTransferBlock_vggv1(channels=64, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
        elif decoder_attn=="v2":
            print("=> build attention module: attn v2")
            self.attn4 = StyleTransferBlock_vggv2(channels=512, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn3 = StyleTransferBlock_vggv2(channels=256, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn2 = StyleTransferBlock_vggv2(channels=128, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn1 = StyleTransferBlock_vggv2(channels=64, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
        else:
            self.decoder_attn = None

        self.decoder4 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.decoder3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.decoder2 = DecoderBlock(input_dim=128, output_dim=64,  hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.decoder1 = DecoderBlock(input_dim=64,  output_dim=3,   hidden_dim=64,  layers=configs[0], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()
    
    def forward(self, skip1, skip2, skip3, skip4, x4):
        y4 = x4
        # style fusion 2
        if self.decoder_attn is not None:
            y4 = self.attn4(y4,y4)
        y3 = self.decoder4(y4, skip=skip4) # (bx512x14x14, bx256x28x28), 14->28
        # style fusion 3
        if self.decoder_attn is not None:
            y3 = self.attn3(y3,y3)
        y2 = self.decoder3(y3, skip=skip3) # (bx256x28x28, bx128x56x56), 28->56
        # style fusion 4
        if self.decoder_attn is not None:
            y2 = self.attn2(y2,y2)
        y1 = self.decoder2(y2, skip=skip2) # (bx128x56x56, bx64x112x112), 56->112
        # style fusion 5
        if self.decoder_attn is not None:
            y1 = self.attn1(y1,y1)
        y0 = self.decoder1(y1,skip=skip1) # (bx128x56x56, bx64x112x112), 112->224
        y0 = self.gate(y0)

        return y0


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(
                    scale_factor=2,          # 放大倍数（如 2 表示宽高都变为 2 倍）
                    mode='bilinear',          # 插值方式：'nearest' | 'bilinear' | 'bicubic' | 'trilinear'
                    align_corners=True      # 是否对齐角点（仅 'bilinear' 和 'bicubic' 有效）
                )
        # upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)
        # self.add_module('0 UpSampling', upsample)
        self.conv = nn.Sequential()
        if layers == 1:
            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)
            self.conv.add_module('1 DecoderLayer', layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.conv.add_module('%d DecoderLayer' % (i+1), layer)

    def forward(self, x, skip):
        if skip is not None:
            x = self.upsample(x) + skip
        else:
            x = self.upsample(x)
        for name, layer in self.conv.named_children():
            x = layer(x)
        return x



class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                # nn.GroupNorm(num_groups=32, num_channels=input_dim, eps=1e-6, affine=True),
                nn.PReLU(),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
    
    def forward(self, x):

        return self.layer(x)



if __name__ == "__main__":

    input1 = torch.randn((1,512,7,7))
    input2 = torch.randn((1,512,14,14))
    input3 = torch.randn((1,256,28,28))
    input4 = torch.randn((1,128,56,56))
    input5 = torch.randn((1,64,112,112))

    configs = get_configs('vgg16')
    print(configs)

    model = VGGDecoder(configs)

    output = model(input1,input2,input3,input4,input5)

    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')
