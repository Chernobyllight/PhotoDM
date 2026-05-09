#!/usr/bin/env python


import os
project_root = os.path.abspath('../..')
import sys
sys.path.append(project_root)

from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import torch

from torchvision.transforms import transforms
from torchvision.transforms.functional import resize

from TRAIN import utils
from TEST import dataloader_val as dataloader
from MODEL import build_transfer

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def crop_border(img_hr, scale):
    b, c, h, w = img_hr.size()

    img_hr = img_hr[:, :, :int(h//scale*scale), :int(w//scale*scale)]

    return img_hr

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Test for auto encoder')


    ### model resume
    parser.add_argument('--resume', default="../../TRAIN/checkpoints/VGGAE.pth", type=str)

    ### model architecture
    parser.add_argument('--arch', default='vgg19', type=str,
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet101', 'resnet152'],
                        help='backbone architechture')
    ### model architecture -> universal
    parser.add_argument('--high_freq_residual', type=str, default="True")  # use high frequency residual?
    parser.add_argument('--pyramid', type=str, default="True")  # use pyramidial high frequency fusion?
    parser.add_argument('--pyramid_version', type=str, default="v2")  # use pyramidial high frequency fusion?
    parser.add_argument('--skips_num', type=int, default=4, choices=[3, 4])  # number of reisuduals
    parser.add_argument('--decoder_attn_version', type=str, default="v2",
                        choices=['v1', 'v2', 'no'])  # attention version
    parser.add_argument('--attn_residual', type=str, default="True")  # if residual in attention block
    parser.add_argument('--use_conv', type=str, default="True")  # if convolution in attention block
    parser.add_argument('--use_selfattn', type=str, default="True")  # if self-attention in attention block
    ### model architecture -> vgg
    parser.add_argument('--encoder_version', type=str, default="v1", choices=['v1',
                                                                              'v2'])  # encoder from torchvision (v2) or else (v1)? v2: only vgg16/19; v1: vgg11/13/16/19
    ### model architecture -> activated when: encoder_version="v2"
    parser.add_argument('--pool_method', type=str, default="max", choices=['average', 'max'])
    ### model architecture -> resnet
    parser.add_argument('--resnet_norm', type=str, default="bn", choices=['gn', 'bn'])


    # style injection setting
    parser.add_argument('--style_condition', type=str, default="efdm") # efdm, hm, id, adain, wct
    parser.add_argument('--kv_injection', type=str, default="t") # if kv injection

    # test data
    parser.add_argument('--val_list_content', default="../../list_IMAGENET/test_content_list.txt", type=str)
    parser.add_argument('--val_list_style', default="../../list_IMAGENET/test_style_list.txt", type=str)
    parser.add_argument('--scale', default=0.25, type=float)

    parser.add_argument('--pretrained_vgg_resume',default=None, type=str)  


    args = parser.parse_args()

    if args.high_freq_residual.lower() in ("true", "yes", "t", "y", "1"):
        args.high_freq_residual = True
    elif args.high_freq_residual.lower() in ("false", "no", "f", "n", "0"):
        args.high_freq_residual = False
    else:
        raise argparse.ArgumentTypeError("need bool")

    if args.pyramid.lower() in ("true", "yes", "t", "y", "1"):
        args.pyramid = True
    elif args.pyramid.lower() in ("false", "no", "f", "n", "0"):
        args.pyramid = False
    else:
        raise argparse.ArgumentTypeError("need bool")

    if args.attn_residual.lower() in ("true", "yes", "t", "y", "1"):
        args.attn_residual = True
    elif args.attn_residual.lower() in ("false", "no", "f", "n", "0"):
        args.attn_residual = False
    else:
        raise argparse.ArgumentTypeError("need bool")
    
    if args.use_conv.lower() in ("true", "yes", "t", "y", "1"):
        args.use_conv = True
    elif args.use_conv.lower() in ("false", "no", "f", "n", "0"):
        args.use_conv = False
    else:
        raise argparse.ArgumentTypeError("need bool")
    
    if args.use_selfattn.lower() in ("true", "yes", "t", "y", "1"):
        args.use_selfattn = True
    elif args.use_selfattn.lower() in ("false", "no", "f", "n", "0"):
        args.use_selfattn = False
    else:
        raise argparse.ArgumentTypeError("need bool")
    
    if args.kv_injection.lower() in ("true", "yes", "t", "y", "1"):
        args.kv_injection = True
    elif args.kv_injection.lower() in ("false", "no", "f", "n", "0"):
        args.kv_injection = False
    else:
        raise argparse.ArgumentTypeError("need bool")
    

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = build_transfer.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    print('=> building the dataloader ...')
    c_loader = dataloader.val_content_loader(args)
    s_loader = dataloader.val_style_loader(args)




    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> transfering ...')

    args.content_output_dir = './content_' + args.style_condition
    content_output_dir = Path(args.content_output_dir)
    if not content_output_dir.exists():
        content_output_dir.mkdir(parents=True, exist_ok=True)


    args.output_dir = './figs_full_' + args.style_condition
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():

        cnt = 0
        for i, (c, _) in enumerate(c_loader):

            for j, (s, _) in enumerate(s_loader):
                cnt += 1

                # print(c.shape, s.shape)
            
                content = c.cuda(non_blocking=True)
                style = s.cuda(non_blocking=True)

                h1, w1 = c.shape[-2:]
                h2, w2 = s.shape[-2:]
                # print(h1, w1)
                # print(h2,w2)
                if h1 >= w1:
                    if h2 >= w2:
                        new_h = h1
                        new_w = w1
                    else:
                        new_h = w1
                        new_w = h1
                elif h1 <= w1:
                    if h2 <= w2:
                        new_h = h1
                        new_w = w1
                    else:
                        new_h = w1
                        new_w = h1
                style = resize(style, (new_h, new_w))

                content = crop_border(content, 16)
                style = crop_border(style, 16)

                print()
                print('---------------------------------------')
                print(content.shape, style.shape)
                output = model.forward(content, style)
                print(output.shape)

                output = transforms.ToPILImage()(output.squeeze().cpu())
                output.save(args.output_dir + "/" + str(cnt) + ".jpg")
                print("stylised save in: ", args.output_dir + "/" + str(cnt) + ".jpg")

                content_save = transforms.ToPILImage()(content.squeeze().cpu())
                content_save.save(args.content_output_dir + "/" + str(cnt) + ".jpg")
                print("content save in: ", args.content_output_dir + "/" + str(cnt) + ".jpg")
                print('---------------------------------------')
                print()


if __name__ == '__main__':

    args = get_args()

    main(args)


