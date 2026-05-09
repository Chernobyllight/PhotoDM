import torch
import numpy as np
import torch.nn as nn
import math 
from xformers.ops import memory_efficient_attention
from skimage.exposure import match_histograms


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# identity
class identity_class(nn.Module):
    def __init__(self,):
        super(identity_class, self).__init__()
    def forward(self, content_feat, style_feat):
        return content_feat


# adain
class adain_class(nn.Module):
    def __init__(self,):
        super(adain_class, self).__init__()
    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


## wct
class wct_class(nn.Module):
    def __init__(self,):
        super(wct_class, self).__init__()

    def forward(self, cf, sf, alpha=1.0):
        cf = cf.squeeze(0)
        sf = sf.squeeze(0)

        return self.wct(cf, sf, alpha=1.0)
    
    def wct(self, cf, sf, beta=None, alpha=1.0):
        # content image whitening
        cf = cf.double()
        c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
        cfv = cf.view(c_channels, -1)  # c x (h x w)

        c_mean = torch.mean(cfv, 1) # perform mean for each row
        c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
        cfv = cfv - c_mean # subtract mean element-wise

        c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)  # construct covariance matrix
        c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

        k_c = c_channels
        for i in range(c_channels):
            if c_e[i] < 0.00001:
                k_c = i
                break
        c_d = (c_e[0:k_c]).pow(-0.5)

        w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
        whitened = torch.mm(w_step2, cfv)

        # style image coloring
        sf = sf.double()
        _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
        sfv = sf.view(c_channels, -1)

        s_mean = torch.mean(sfv, 1)
        s_mean = s_mean.unsqueeze(1).expand_as(sfv)
        sfv = sfv - s_mean

        s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
        s_u, s_e, s_v = torch.svd(s_covm, some=False)

        s_k = c_channels # same number of channels ad content features
        for i in range(c_channels):
            if s_e[i] < 0.00001:
                s_k = i
                break
        s_d = (s_e[0:s_k]).pow(0.5)

        c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
        colored = torch.mm(c_step2, whitened)

        cs0_features = colored + s_mean.resize_as_(colored)
        cs0_features = cs0_features.view_as(cf)

        # additional style coloring
        if beta:
            sf = s1f
            sf = sf.double()
            _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
            sfv = sf.view(c_channels, -1)

            s_mean = torch.mean(sfv, 1)
            s_mean = s_mean.unsqueeze(1).expand_as(sfv)
            sfv = sfv - s_mean

            s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
            s_u, s_e, s_v = torch.svd(s_covm, some=False)

            s_k = c_channels
            for i in range(c_channels):
                if s_e[i] < 0.00001:
                    s_k = i
                    break
            s_d = (s_e[0:s_k]).pow(0.5)

            c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
            c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
            colored = torch.mm(c_step2, whitened)

            cs1_features = colored + s_mean.resize_as_(colored)
            cs1_features = cs1_features.view_as(cf)

            target_features = beta * cs0_features + (1.0 - beta) * cs1_features
        else:
            target_features = cs0_features

        ccsf = alpha * target_features + (1.0 - alpha) * cf
        return ccsf.float().unsqueeze(0)



## EFDM
class exact_feature_distribution_matching_class(nn.Module):
    def __init__(self,):
        super(exact_feature_distribution_matching_class, self).__init__()
    
    def forward(self, content_feat, style_feat):
        assert (content_feat.size() == style_feat.size())
        B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
        value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
        value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
        inverse_index = index_content.argsort(-1)
        new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())

        return new_content.view(B, C, W, H)




## HM
class histogram_matching_class(nn.Module):
    def __init__(self,):
        super(histogram_matching_class, self).__init__()
    
    def forward(self, content_feat, style_feat):
        assert (content_feat.size() == style_feat.size())
        B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
        x_view = content_feat.view(-1, W,H)
        image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
                                    np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2)),
                                    # multichannel=True,
                                    channel_axis=-1,
                                    )
        image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
        return content_feat + (image1_temp - content_feat).detach()







def Normalize(num_groups, in_channels):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)









class MultiheadAttentionv4(nn.Module):
    def __init__(self, in_channels, n_heads,residual=False, attn_scale=1, norm_style_key=False):
        super(MultiheadAttentionv4, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.residual = residual
        self.attn_scale = attn_scale
        self.hid_dim = self.in_channels * self.attn_scale
        self.norm_style_key = norm_style_key

        # 强制 in_channels 必须整除 h
        assert in_channels % n_heads == 0
        self.norm = Normalize(num_groups=32, in_channels=in_channels)
        # self.norm = nn.BatchNorm2d(input_channels)
        # self.attn = torch.nn.MultiheadAttention(embed_dim=self.in_channels, num_heads=self.n_heads, batch_first=True)

        self.w_q = torch.nn.Conv2d(in_channels,
                                 in_channels * attn_scale,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.w_k = torch.nn.Conv2d(in_channels,
                                 in_channels * attn_scale,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.w_v = torch.nn.Conv2d(in_channels,
                                 in_channels * attn_scale,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels * attn_scale,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        # 缩放
        # self.scale = torch.sqrt(torch.FloatTensor([in_channels // n_heads])).cuda()
        self.scale = math.sqrt(in_channels // n_heads)

    def forward(self, query, key, tau=1.0):
        q_residual = query
        bsz = query.shape[0]

        h_ = query
        h_ = self.norm(h_)
        q = self.w_q(h_)

        if (key is not None):
            if self.norm_style_key:
                key_norm = self.norm(key)
                k = self.w_k(key_norm)
                v = self.w_v(key_norm)
            else:
                k = self.w_k(key)
                v = self.w_v(key)
        else:
            k = self.w_k(h_)
            v = self.w_v(h_)
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1).contiguous()    # b,hw,c

        bk,ck,hk,wk = k.shape
        k = k.reshape(bk,ck,hk*wk) # b,c,hw
        k = k.permute(0,2,1).contiguous()    # b,hw,c
        v = v.reshape(bk,ck,hk*wk) # b,c,hw
        v = v.permute(0,2,1).contiguous()    # b,hw,c

        Q = q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).contiguous()
        K = k.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).contiguous()
        V = v.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).contiguous()

        Q = Q * tau
        output = memory_efficient_attention(Q,K,V)

        output = output.contiguous()
        output = output.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)) # b,hw,c
        output = output.permute(0,2,1).contiguous()
        output = output.reshape(b,c,h,w)
        output = self.proj_out(output)


        if self.residual:
            return output + q_residual
        else:
            return output




if __name__ == "__main__":

    # import torch

    # x = torch.tensor([1.0], requires_grad=True)  # 原张量需要梯度
    # y = x.clone()  # 深复制
    # print(y.requires_grad)
    # exit(123)

    c = torch.randn((2,64,128,128)).cuda()
    s = torch.randn((2,64,128,128)).cuda()

    kv_in = histogram_matching_class()
    print(kv_in(c,s).shape)
    exit(123)

    # infer content self attn
    print('infer content self attn')
    model = MultiheadAttention(in_channels=64, n_heads=4,residual=True,attn_scale=1,norm_style_key=False).cuda()
    output = model(c,None)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    model = MultiheadAttentionv2(in_channels=64, n_heads=4,residual=True,norm_style_key=False).cuda()
    output = model(c,None)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    # infer content-style kv injection (cross attn)
    print('---------------------------')
    print('infer content-style kv injection (cross attn): multiha v1')
    model = MultiheadAttention(in_channels=64, n_heads=4,residual=True,attn_scale=1,norm_style_key=False).cuda()
    output = model(c,s)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    model = MultiheadAttention(in_channels=64, n_heads=4,residual=True,attn_scale=1,norm_style_key=True).cuda()
    output = model(c,s)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')
    print('---------------------------')
    print('infer content-style kv injection (cross attn): multiha v2')
    model = MultiheadAttentionv2(in_channels=64, n_heads=4,residual=True,norm_style_key=False).cuda()
    output = model(c,s)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')

    model = MultiheadAttentionv2(in_channels=64, n_heads=4,residual=True,norm_style_key=True).cuda()
    output = model(c,s)
    print(output.shape)
    print('# net parameters:', sum(param.numel() for param in model.parameters()), '\n')
    print('---------------------------')

    