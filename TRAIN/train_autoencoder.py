#!/usr/bin/env python

import os
project_root = os.path.abspath('..')
import sys
sys.path.append(project_root)

from pathlib import Path
import time
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from TRAIN import utils
from DATA import dataloader
from MODEL import build_ae


# torch.backends.cuda.enable_flash_sdp(True)

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')

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
    





    parser.add_argument('--train_list', default="../list_IMAGENET/PST_list.txt", type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--load_resume', default=None, type=str) # continue training 
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')


    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='./checkpoints', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=1, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=0,
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10007', type=str,
                    help='url used to set up distributed training')                                            

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



    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)
    
def main_worker(gpu, args):
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))
    model = build_ae.BuildAutoEncoder(args) 
    if args.load_resume is not None:
        print('=> continue training, loading pth from {} ...'.format(args.load_resume))
        utils.load_dict(args.load_resume, model)
    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('=> num of trainable params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    if args.rank == 0:
        print('=> building the oprimizer ...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.rank == 0:
        print('=> building the dataloader ...')
    train_loader = dataloader.train_loader(args)

    if args.rank == 0:
        print('=> building the criterion ...')
    criterion = nn.MSELoss()

    if args.rank == 0:
        output_dir = Path(args.pth_save_fold)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)


    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)

        if args.parallel != 0:
            train_loader.sampler.set_epoch(epoch)
            
        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, '{}.pth'.format(str(epoch).zfill(3)))
            )
            
            print(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch+1))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1
         
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        loss.backward()
        # 限制梯度范数
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # update weights
        optimizer.step()

        # syn for logging
        torch.cuda.synchronize()

        # record loss
        losses.update(loss.item(), input.size(0))          

        # measure elapsed time
        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)

            # state_dict = model.state_dict()
            # torch.save(
            #     {
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': state_dict,
            #         'optimizer' : optimizer.state_dict(),
            #     },
            #     os.path.join(args.pth_save_fold, '{}.pth'.format(str(epoch).zfill(3)))
            # )
            # print(' : save pth for epoch {}'.format(epoch + 1))
            # exit(123)
        





if __name__ == '__main__':

    args = get_args()

    main(args)


