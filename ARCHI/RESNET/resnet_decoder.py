import torch
import torch.nn as nn
from ARCHI.TRANSFORM.attn_trans import StyleTransferBlock_vggv1, StyleTransferBlock_vggv2



class ResNetDecoder(nn.Module):

    def __init__(self, configs, 
                decoder_attn="v2", attn_residual=True, use_conv=True, use_selfattn=True, resnet_norm="gn"):

        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.decoder_attn = decoder_attn
        if decoder_attn=="v1":
            print("build attention module: attn v1")
            self.attn4 = StyleTransferBlock_vggv1(channels=512, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn3 = StyleTransferBlock_vggv1(channels=256, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn2 = StyleTransferBlock_vggv1(channels=128, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn1 = StyleTransferBlock_vggv1(channels=64, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
        elif decoder_attn=="v2":
            print("build attention module: attn v2")
            self.attn4 = StyleTransferBlock_vggv2(channels=512, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn3 = StyleTransferBlock_vggv2(channels=256, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn2 = StyleTransferBlock_vggv2(channels=128, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
            self.attn1 = StyleTransferBlock_vggv2(channels=64, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn, n_heads=8, norm_style_key=True)
        else:
            self.decoder_attn = None
        

        self.conv4 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[3], resnet_norm=resnet_norm)
        self.conv3 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[2], resnet_norm=resnet_norm)
        self.conv2 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[1], resnet_norm=resnet_norm)
        self.conv1 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[0], resnet_norm=resnet_norm)

        self.conv0 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        self.gate = nn.Sigmoid()
    
    def forward(self,x4):
        if self.decoder_attn is not None:
            x4 = self.attn4(x4,x4)
        x3 = self.conv4(x4)

        if self.decoder_attn is not None:
            x3 = self.attn3(x3,x3)
        x2 = self.conv3(x3)

        if self.decoder_attn is not None:
            x2 = self.attn2(x2,x2)
        x1 = self.conv2(x2)

        if self.decoder_attn is not None:
            x1 = self.attn1(x1,x1)
        x0 = self.conv1(x1)
        img_temp = self.conv0(x0)
        x = self.gate(img_temp)
        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers, resnet_norm):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True,resnet_norm=resnet_norm)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False,resnet_norm=resnet_norm)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample, resnet_norm):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
                nn.PReLU(),
                # nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                nn.Conv2d(in_channels=hidden_channels, out_channels=4 * output_channels, kernel_size=3, stride=1, padding=1), 
                nn.PixelShuffle(2)    
            )
        else:
            self.weight_layer2 = nn.Sequential(
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
                nn.PReLU(),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            )

        if upsample:
            self.upsample = nn.Sequential(
                # nn.BatchNorm2d(num_features=hidden_channels),
                nn.GroupNorm(num_groups=32, num_channels=hidden_channels) if resnet_norm == "gn" else nn.BatchNorm2d(num_features=hidden_channels),
                nn.PReLU(),
                # nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1)   
                nn.Conv2d(in_channels=hidden_channels, out_channels=4 * output_channels, kernel_size=3, stride=1, padding=1), 
                nn.PixelShuffle(2)  
            )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x
