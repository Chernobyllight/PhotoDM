import torch
import torch.nn as nn

from collections import OrderedDict


from ARCHI.RESNET.resnet_ae import ResNet_AutoEncoder
from ARCHI.TRANSFORM.functions import exact_feature_distribution_matching_class, wct_class, adain_class, histogram_matching_class, identity_class

class RESNETAE_PST(nn.Module):

    def __init__(self, 
                    resnet_type='resnet34', decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=True,resnet_norm="gn",

                    RESNET_resume=None, style_condition="efdm", kv_injection=True
                    ):

        super(RESNETAE_PST, self).__init__()
        self.AE_base = ResNet_AutoEncoder(resnet_type=resnet_type, 
        decoder_attn=decoder_attn,attn_residual=attn_residual, 
        use_conv=use_conv, use_selfattn=use_selfattn,resnet_norm=resnet_norm)

        if RESNET_resume != None:
            checkpoint = torch.load(RESNET_resume)
            # 去除'module.'前缀
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    name = k[7:]  # 去掉 'module.'
                else:
                    name = k
                new_state_dict[name] = v

            self.AE_base.load_state_dict(new_state_dict, strict=True)
            print("=> successful load autoencoder")

        if style_condition == "efdm":
            self.sc_in = exact_feature_distribution_matching_class()
        elif style_condition == "wct":
            self.sc_in = wct_class()
        elif style_condition == "adain":
            self.sc_in = adain_class()
        elif style_condition == "hm":
            self.sc_in = histogram_matching_class()
        elif style_condition == "id":
            self.sc_in = identity_class()

        self.kv_injection = kv_injection
        if self.kv_injection:
            print("=> transfer with kv injection")



    def forward(self, content, style,tau=1.0, lambda_1=0.0):
        self.AE_base.freeze()

        fc4 = self.AE_base.ae_encoder(content)
        fs4 = self.AE_base.ae_encoder(style)
        
        # fusion round1:
        fcs4 = self.sc_in(fc4, fs4) # optional
        if self.kv_injection:
            fcs4 = self.AE_base.ae_decoder.attn4(fcs4,fs4,tau) # optional
        else:
            fcs4 = self.AE_base.ae_decoder.attn4(fcs4,fcs4,tau)
        fs4 = self.AE_base.ae_decoder.attn4(fs4,fs4,1)
        fc4 = self.AE_base.ae_decoder.attn4(fc4,fc4,1)
        fcs4 = self.sc_in(fcs4, fs4) # optional
        fcs4 = lambda_1 * fc4 + (1-lambda_1) * fcs4 # optional
        ofcs3 = self.AE_base.ae_decoder.conv4(fcs4)
        ofs3 = self.AE_base.ae_decoder.conv4(fs4)
        ofc3 = self.AE_base.ae_decoder.conv4(fc4)

        # fusion round2:
        ofcs3 = self.sc_in(ofcs3, ofs3) # optional
        if self.kv_injection:
            fcs3 = self.AE_base.ae_decoder.attn3(ofcs3, ofs3,tau) # optional
        else:
            fcs3 = self.AE_base.ae_decoder.attn3(ofcs3,ofcs3,tau)
        ofs3 = self.AE_base.ae_decoder.attn3(ofs3,ofs3,1)
        ofc3 = self.AE_base.ae_decoder.attn3(ofc3,ofc3,1)
        fcs3 = self.sc_in(fcs3, ofs3) # optional
        fcs3 = lambda_1 * ofc3+ (1-lambda_1) * fcs3 # optional
        # decoder4:
        ofcs2 = self.AE_base.ae_decoder.conv3(fcs3)
        ofs2 = self.AE_base.ae_decoder.conv3(ofs3)
        ofc2 = self.AE_base.ae_decoder.conv3(ofc3)


        # fusion round3:
        ofcs2 = self.sc_in(ofcs2, ofs2) # optional
        if self.kv_injection:
            fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofs2,tau) # optional
        else:
            fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofcs2,tau)
        ofs2 = self.AE_base.ae_decoder.attn2(ofs2,ofs2,1)
        ofc2 = self.AE_base.ae_decoder.attn2(ofc2,ofc2,1)
        fcs2 = self.sc_in(fcs2, ofs2) # optional
        fcs2 = lambda_1 * ofc2+ (1-lambda_1) * fcs2 # optional
        ofcs1 = self.AE_base.ae_decoder.conv2(fcs2)
        ofs1 = self.AE_base.ae_decoder.conv2(ofs2)
        ofc1 = self.AE_base.ae_decoder.conv2(ofc2)

        # fusion round2:
        ofcs1 = self.sc_in(ofcs1, ofs1) # optional
        if self.kv_injection:
            fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofs1,tau) # optional
        else:
            fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofcs1,tau)
        ofs1 = self.AE_base.ae_decoder.attn1(ofs1,ofs1,1)
        ofc1 = self.AE_base.ae_decoder.attn1(ofc1,ofc1,1)
        fcs1 = self.sc_in(fcs1, ofs1) # optional
        fcs1 = lambda_1 * ofc1+ (1-lambda_1) * fcs1 # optional
        fcs0 = self.AE_base.ae_decoder.conv1(fcs1)
        ofs0 = self.AE_base.ae_decoder.conv1(ofs1)
        ofc0 = self.AE_base.ae_decoder.conv1(ofc1)

        fcs0 = self.sc_in(fcs0, ofs0)

        img_fcs_temp = self.AE_base.ae_decoder.conv0(fcs0)
        img_ofs0_temp = self.AE_base.ae_decoder.conv0(ofs0)

        img_fcs_temp = self.sc_in(img_fcs_temp, img_ofs0_temp)

        ofcs_img = self.AE_base.ae_decoder.gate(img_fcs_temp)
        return ofcs_img

    
    def freeze_model(self, model):
        for (name, param) in model.named_parameters():
            param.requires_grad = False
        
        return model
        
