import torch.nn as nn
import torch.nn.parallel as parallel

from ARCHI.VGG.vgg_transfer import VGGAE_PST
from ARCHI.RESNET.resnet_transfer import RESNETAE_PST

def BuildAutoEncoder(args):

    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        model = VGGAE_PST(vgg_type=args.arch, 
            enable_bn_en=True, enable_bn_de=True ,
            high_freq_residual=args.high_freq_residual,pyramid=args.pyramid, pyramid_version=args.pyramid_version, skips=args.skips_num, 
            decoder_attn=args.decoder_attn_version,attn_residual=args.attn_residual, use_conv=args.use_conv, use_selfattn=args.use_selfattn,
            
            encoder_version=args.encoder_version, args=args,
            
            VGG_resume=args.resume, style_condition=args.style_condition, kv_injection=args.kv_injection
            )

    elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        model = RESNETAE_PST(resnet_type=args.arch, 
        decoder_attn=args.decoder_attn_version,attn_residual=args.attn_residual, 
        use_conv=args.use_conv, use_selfattn=args.use_selfattn, resnet_norm=args.resnet_norm,
        
        RESNET_resume=args.resume, style_condition=args.style_condition, kv_injection=args.kv_injection
        )
    
    else:
        return None
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu
                    )  
    
    else:
        model = nn.DataParallel(model).cuda()

    return model