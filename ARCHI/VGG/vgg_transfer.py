import torch
import torch.nn as nn

from collections import OrderedDict


from ARCHI.VGG.vgg_ae import VGG_AutoEncoder
from ARCHI.TRANSFORM.functions import exact_feature_distribution_matching_class, wct_class, adain_class, histogram_matching_class, identity_class


class VGGAE_PST(nn.Module):

    def __init__(self, 
    vgg_type='vgg19',  enable_bn_en=True, enable_bn_de=True ,
    high_freq_residual=True,pyramid=True, pyramid_version="v1", skips=3, 
    decoder_attn="v2",attn_residual=True, use_conv=True, use_selfattn=True,

    encoder_version="v2", args=None,
    
    VGG_resume=None, style_condition="efdm", kv_injection=True):

        super(VGGAE_PST, self).__init__()
        self.AE_base = VGG_AutoEncoder(
            vgg_type=vgg_type, 
            enable_bn_en=enable_bn_en, enable_bn_de=enable_bn_de,
            high_freq_residual=high_freq_residual, pyramid=pyramid,pyramid_version=pyramid_version, skips=skips, 
            decoder_attn=decoder_attn, attn_residual=attn_residual, use_conv=use_conv, use_selfattn=use_selfattn,

            encoder_version=encoder_version, args=args
        )

        if VGG_resume != None:
            checkpoint = torch.load(VGG_resume)
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
        # skip2, skip3, skip4, skip5, x5
        skip_fc1, skip_fc2, skip_fc3, skip_fc4, fc4 = self.AE_base.ae_encoder(content)
        skip_fs1, skip_fs2, skip_fs3, skip_fs4, fs4 = self.AE_base.ae_encoder(style)

        # fusion round1:
        fc4 = self.sc_in(fc4, fs4) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs4 = self.AE_base.ae_decoder.attn4(fc4,fs4,tau) # optional
            else:
                fcs4 = self.AE_base.ae_decoder.attn4(fc4,fc4,tau)
            fs4 = self.AE_base.ae_decoder.attn4(fs4,fs4,1)
            fc4 = self.AE_base.ae_decoder.attn4(fc4,fc4,1)
        else:
            fcs4 = fc4
            fs4 = fs4
            fc4 = fc4
        ofcs3 = self.upsamplev1(self.AE_base.ae_decoder.decoder4.upsample, fcs4, skip_fc4)
        ofs3 = self.upsamplev1(self.AE_base.ae_decoder.decoder4.upsample, fs4, skip_fs4)
        ofc3 = self.upsamplev1(self.AE_base.ae_decoder.decoder4.upsample, fc4, skip_fc4)
        ofcs3 = self.sc_in(ofcs3, ofs3) # optional
        ofcs3 = lambda_1 * ofc3+ (1-lambda_1) * ofcs3 # optional
        ofcs3 = self.AE_base.ae_decoder.decoder4.conv(ofcs3)
        ofs3 = self.AE_base.ae_decoder.decoder4.conv(ofs3)
        ofc3 = self.AE_base.ae_decoder.decoder4.conv(ofc3)


        # fusion round2:
        ofcs3 = self.sc_in(ofcs3, ofs3) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs3 = self.AE_base.ae_decoder.attn3(ofcs3, ofs3,tau) # optional
            else:
                fcs3 = self.AE_base.ae_decoder.attn3(ofcs3,ofcs3,tau)
            ofs3 = self.AE_base.ae_decoder.attn3(ofs3,ofs3,1)
            ofc3 = self.AE_base.ae_decoder.attn3(ofc3,ofc3,1)
        else:
            fcs3 = ofcs3
            ofs3 = ofs3
            ofc3 = ofc3
        ofcs2 = self.upsamplev1(self.AE_base.ae_decoder.decoder3.upsample, fcs3, skip_fc3)
        ofs2 = self.upsamplev1(self.AE_base.ae_decoder.decoder3.upsample, ofs3, skip_fs3)
        ofc2 = self.upsamplev1(self.AE_base.ae_decoder.decoder3.upsample, ofc3, skip_fc3)
        ofcs2 = self.sc_in(ofcs2, ofs2) # optional
        ofcs2 = lambda_1 * ofc2+ (1-lambda_1) * ofcs2 # optional
        ofcs2 = self.AE_base.ae_decoder.decoder3.conv(ofcs2)
        ofs2 = self.AE_base.ae_decoder.decoder3.conv(ofs2)
        ofc2 = self.AE_base.ae_decoder.decoder3.conv(ofc2)


        # fusion round3:
        ofcs2 = self.sc_in(ofcs2, ofs2) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofs2,tau) # optional
            else:
                fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofcs2,tau)
            ofs2 = self.AE_base.ae_decoder.attn2(ofs2,ofs2,1)
            ofc2 = self.AE_base.ae_decoder.attn2(ofc2,ofc2,1)
        else:
            fcs2 = ofcs2
            ofs2 = ofs2
            ofc2 = ofc2
        ofcs1 = self.upsamplev1(self.AE_base.ae_decoder.decoder2.upsample, fcs2, skip_fc2)
        ofs1 = self.upsamplev1(self.AE_base.ae_decoder.decoder2.upsample, ofs2, skip_fs2)
        ofc1 = self.upsamplev1(self.AE_base.ae_decoder.decoder2.upsample, ofc2, skip_fc2)
        ofcs1 = self.sc_in(ofcs1, ofs1) # optional
        ofcs1 = lambda_1 * ofc1+ (1-lambda_1) * ofcs1 # optional
        ofcs1 = self.AE_base.ae_decoder.decoder2.conv(ofcs1)
        ofs1 = self.AE_base.ae_decoder.decoder2.conv(ofs1)
        ofc1 = self.AE_base.ae_decoder.decoder2.conv(ofc1)

        # fusion round2:
        ofcs1 = self.sc_in(ofcs1, ofs1) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofs1,tau) # optional
            else:
                fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofcs1,tau)
            ofs1 = self.AE_base.ae_decoder.attn1(ofs1,ofs1,1)
            ofc1 = self.AE_base.ae_decoder.attn1(ofc1,ofc1,1)
        else:
            fcs1 = ofcs1
            ofs1 = ofs1
            ofc1 = ofc1
        ofcs0 = self.upsamplev1(self.AE_base.ae_decoder.decoder1.upsample, fcs1, skip_fc1)
        ofs0 = self.upsamplev1(self.AE_base.ae_decoder.decoder1.upsample, ofs1, skip_fs1)
        ofc0 = self.upsamplev1(self.AE_base.ae_decoder.decoder1.upsample, ofc1, skip_fc1)
        ofcs0 = self.sc_in(ofcs0, ofs0) # optional
        ofcs0 = lambda_1 * ofc0 + (1-lambda_1) * ofcs0 # optional
        ofcs_img = self.AE_base.ae_decoder.decoder1.conv(ofcs0)
        ofs0 = self.AE_base.ae_decoder.decoder1.conv(ofs0)
        ofc0 = self.AE_base.ae_decoder.decoder1.conv(ofc0)

        ofcs_img = self.sc_in(ofcs_img, ofs0) # optional
        # ofcs_img = lambda_1 * ofc0 + (1-lambda_1) * ofcs_img # optional

        ofcs_img = self.AE_base.ae_decoder.gate(ofcs_img)
        # ofcs_img = self.AE_base.ae_decoder.gate(ofs0)

        return ofcs_img

    def forwardv1(self, content, style,tau=1.0):
        self.AE_base.freeze()
        # skip2, skip3, skip4, skip5, x5
        skip_fc1, skip_fc2, skip_fc3, skip_fc4, fc4 = self.AE_base.ae_encoder(content)
        skip_fs1, skip_fs2, skip_fs3, skip_fs4, fs4 = self.AE_base.ae_encoder(style)

        # skip_fs1, skip_fs2, skip_fs3, skip_fs4 = 0,0,0,0
        # fs4 = torch.zeros_like(fs4)
        # skip_fs1, skip_fs2, skip_fs3, skip_fs4 = 0,0,0,0

        # fusion round1:
        fc4 = self.sc_in(fc4, fs4) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs4 = self.AE_base.ae_decoder.attn4(fc4,fs4,tau) # optional
            else:
                fcs4 = self.AE_base.ae_decoder.attn4(fc4,fc4,tau)
            fs4 = self.AE_base.ae_decoder.attn4(fs4,fs4,1)
        else:
            fcs4 = fc4
            fs4 = fs4
        ofcs3 = self.upsamplev1(self.AE_base.ae_decoder.decoder4.upsample, fcs4, skip_fc4)
        ofs3 = self.upsamplev1(self.AE_base.ae_decoder.decoder4.upsample, fs4, skip_fs4)
        ofcs3 = self.sc_in(ofcs3, ofs3) # optional
        ofcs3 = self.AE_base.ae_decoder.decoder4.conv(ofcs3)
        ofs3 = self.AE_base.ae_decoder.decoder4.conv(ofs3)


        # fusion round2:
        ofcs3 = self.sc_in(ofcs3, ofs3) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs3 = self.AE_base.ae_decoder.attn3(ofcs3, ofs3,tau) # optional
            else:
                fcs3 = self.AE_base.ae_decoder.attn3(ofcs3,ofcs3,tau)
            ofs3 = self.AE_base.ae_decoder.attn3(ofs3,ofs3,1)
        else:
            fcs3 = ofcs3
            ofs3 = ofs3
        ofcs2 = self.upsamplev1(self.AE_base.ae_decoder.decoder3.upsample, fcs3, skip_fc3)
        ofs2 = self.upsamplev1(self.AE_base.ae_decoder.decoder3.upsample, ofs3, skip_fs3)
        ofcs2 = self.sc_in(ofcs2, ofs2) # optional
        ofcs2 = self.AE_base.ae_decoder.decoder3.conv(ofcs2)
        ofs2 = self.AE_base.ae_decoder.decoder3.conv(ofs2)


        # fusion round3:
        ofcs2 = self.sc_in(ofcs2, ofs2) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofs2,tau) # optional
            else:
                fcs2 = self.AE_base.ae_decoder.attn2(ofcs2,ofcs2,tau)
            ofs2 = self.AE_base.ae_decoder.attn2(ofs2,ofs2,1)
        else:
            fcs2 = ofcs2
            ofs2 = ofs2
        ofcs1 = self.upsamplev1(self.AE_base.ae_decoder.decoder2.upsample, fcs2, skip_fc2)
        ofs1 = self.upsamplev1(self.AE_base.ae_decoder.decoder2.upsample, ofs2, skip_fs2)
        ofcs1 = self.sc_in(ofcs1, ofs1) # optional
        ofcs1 = self.AE_base.ae_decoder.decoder2.conv(ofcs1)
        ofs1 = self.AE_base.ae_decoder.decoder2.conv(ofs1)

        # fusion round2:
        ofcs1 = self.sc_in(ofcs1, ofs1) # optional
        if self.AE_base.ae_decoder.decoder_attn != None:
            if self.kv_injection:
                fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofs1,tau) # optional
            else:
                fcs1 = self.AE_base.ae_decoder.attn1(ofcs1,ofcs1,tau)
            ofs1 = self.AE_base.ae_decoder.attn1(ofs1,ofs1,1)
        else:
            fcs1 = ofcs1
            ofs1 = ofs1
        ofcs0 = self.upsamplev1(self.AE_base.ae_decoder.decoder1.upsample, fcs1, skip_fc1)
        ofs0 = self.upsamplev1(self.AE_base.ae_decoder.decoder1.upsample, ofs1, skip_fs1)
        ofcs0 = self.sc_in(ofcs0, ofs0) # optional
        ofcs_img = self.AE_base.ae_decoder.decoder1.conv(ofcs0)
        ofs0 = self.AE_base.ae_decoder.decoder1.conv(ofs0)

        ofcs_img = self.sc_in(ofcs_img, ofs0) # optional

        ofcs_img = self.AE_base.ae_decoder.gate(ofcs_img)
        # ofcs_img = self.AE_base.ae_decoder.gate(ofs0)

        return ofcs_img


    def upsamplev1(self, upsampler, input_x, skip):
        if skip != None:
            output = upsampler(input_x) + skip
        else:
            output = upsampler(input_x)

        return output


    def freeze_model(self, model):
        for (name, param) in model.named_parameters():
            param.requires_grad = False
        
        return model
        
if __name__ == "__main__":

    c = torch.randn((2,3,512,512)).cuda()
    s = torch.randn((2,3,512,512)).cuda()

    model = VGG_AutoEncoder_ST(vgg_type='vgg16', enable_bn_en=True, enable_bn_de=True, kernel_size=3,pyramid=False).cuda()

    # output = model.forward_train(c,s)
    output = model.forward(c,s)

    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')
    # print(output.device)
    # for parameter in model.parameters():
    #     print(parameter.requires_grad)
