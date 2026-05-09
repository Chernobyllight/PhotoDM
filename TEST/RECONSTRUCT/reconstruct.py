#!/usr/bin/env python


import os
project_root = os.path.abspath('../..')
import sys
sys.path.append(project_root)



import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import torch

from torchvision.transforms import transforms

from TRAIN import utils
from TEST import dataloader_val as dataloader
from MODEL import build_ae

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='reconstruct images')





    ### model resume
    parser.add_argument('--resume',default="../../TRAIN/checkpoints/011.pth", type=str)


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







    parser.add_argument('--val_list', default="../../list_IMAGENET/PST_recon_list.txt", type=str)


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
    
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = build_ae.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    print('=> building the dataloader ...')
    train_loader = dataloader.val_loader(args)

    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)

            input = transforms.ToPILImage()(input.squeeze().cpu())
            output = transforms.ToPILImage()(output.squeeze().cpu())

            plt.subplot(8,16,2*i+1, xticks=[], yticks=[])
            plt.imshow(input)

            plt.subplot(8,16,2*i+2, xticks=[], yticks=[])
            plt.imshow(output)

            if i == 63:
                break

    plt.savefig('reconstruction.jpg')

if __name__ == '__main__':

    args = get_args()

    main(args)


