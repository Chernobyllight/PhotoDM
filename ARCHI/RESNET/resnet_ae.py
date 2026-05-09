
import torch
import torch.nn as nn
from ARCHI.RESNET.resnet_encoder import ResNetEncoder
from ARCHI.RESNET.resnet_decoder import ResNetDecoder


class ResNet_AutoEncoder(nn.Module):

    def __init__(self, resnet_type, decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=True,resnet_norm='gn'):

        super(ResNet_AutoEncoder, self).__init__()
        print('init ResNet AE')
        if resnet_type == 'resnet18':
            configs = [2, 2, 2, 2]
        elif resnet_type == 'resnet34':
            configs = [3, 4, 6, 3]
        elif resnet_type == 'resnet101':
            configs = [3, 4, 23, 3]
        elif resnet_type == 'resnet152':
            configs = [3, 8, 36, 3]
        else:
            raise ValueError("Undefined model")
        

        self.ae_encoder = ResNetEncoder(configs=configs, resnet_norm=resnet_norm)
        self.ae_decoder = ResNetDecoder(configs=configs, 
        decoder_attn=decoder_attn,attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn,resnet_norm=resnet_norm)

    def forward(self, img):
        x4 = self.ae_encoder(img)
        recons_img = self.ae_decoder(x4)
        return recons_img

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False




if __name__ == "__main__":

    inputimg = torch.randn((2,3,256,256)).cuda()

    model = ResNet_AutoEncoder(resnet_type='resnet34',decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=False
    ).cuda()
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    output = model(inputimg)

    print(output.shape)
    print(output.device)


