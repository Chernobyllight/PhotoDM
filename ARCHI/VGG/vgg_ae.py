
import torch
import torch.nn as nn
from ARCHI.VGG.vgg_encoder import VGGEncoder
from ARCHI.VGG.vgg_encoderv2 import VGGEncoderv2
from ARCHI.VGG.vgg_decoder import VGGDecoder


class VGG_AutoEncoder(nn.Module):

    def __init__(self, vgg_type, args=None, enable_bn_en=True, enable_bn_de=True ,
    high_freq_residual=True,pyramid=True, pyramid_version="v1", skips=3, 
    decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=True,

    encoder_version='v1',
    ):
        super(VGG_AutoEncoder, self).__init__()
        print('init VGG AE')

        if vgg_type == 'vgg11':
            configs = [1, 1, 2, 2, 2]
        elif vgg_type == 'vgg13':
            configs = [2, 2, 2, 2, 2]
        elif vgg_type == 'vgg16':
            configs = [2, 2, 3, 3, 3]
        elif vgg_type == 'vgg19':
            configs = [2, 2, 4, 4, 4]
        else:
            raise ValueError("Undefined model")
        
        if encoder_version == "v1":
            self.ae_encoder = VGGEncoder(configs=configs, enable_bn=enable_bn_en, high_freq_residual=high_freq_residual, pyramid=pyramid, pyramid_version=pyramid_version, skips=skips)
        elif encoder_version == "v2":
            self.ae_encoder = VGGEncoderv2(vgg_type=vgg_type, pool_method=args.pool_method,
             high_freq_residual=high_freq_residual, pyramid=pyramid, skips=skips, pretrained_vgg_resume=args.pretrained_vgg_resume,pyramid_version=pyramid_version)

        self.ae_decoder = VGGDecoder(configs=configs, enable_bn=enable_bn_de, 
        decoder_attn=decoder_attn,
        attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn) 

    def forward(self, img):
        skip1, skip2, skip3, skip4, x4 = self.ae_encoder(img)
        recons_img = self.ae_decoder(skip1, skip2, skip3, skip4, x4)
        return recons_img

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False






if __name__ == "__main__":

    inputimg = torch.randn((2,3,256,256)).cuda()

    model = VGG_AutoEncoder(vgg_type='vgg16', enable_bn_en=True, enable_bn_de=True,pyramid=True,skips=4,
    decoder_attn="v2",attn_residual=False, use_conv=True, use_selfattn=False
    ).cuda()
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    output = model(inputimg)

    print(output.shape)
    print(output.device)

