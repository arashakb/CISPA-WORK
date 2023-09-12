import os
#os.environ['CUDA_HOME'] = "/usr/local/cuda"
#os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'  compile PSANet has some trouble
import json
import argparse
import torch
import dataloaders
import models
import torch.nn as nn
import torch.nn.functional as F
from util import METRICS, setup_seed
from whitebox_lib import get_adv_examples_CRPGD
from torchvision.utils import save_image
import torchvision.transforms as T 
from PIL import Image 
from util import save_target, save_images

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])



CNT = 0


def main(args, config):
    
    setup_seed(100)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    num_classes = val_loader.dataset.num_classes
    name_arch = config['arch']['type']
    name_data = config['train_loader']['type']
    args.binname = config['val_loader']['binname']
    if not os.path.exists('log'):
        os.mkdir('log')
    args.log = 'log/log_{}_{}_{}_{}_{}_{}.txt'.format(name_data, name_arch,
                                                      args.norm, args.eps,
                                                      args.a, args.b)
    with open(args.log, 'w') as f:
        print(config, file=f)
        f.close()
    model = getattr(models, config['arch']['type'])(num_classes,
                                                    **config['arch']['args'])

    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    checkpoint = torch.load(args.binname, map_location=torch.device('cpu'))

    if checkpoint.get('state_dict') != None:
        checkpoint = checkpoint["state_dict"]
    model.set_mean_std(val_loader.MEAN, val_loader.STD)
    model.eval()
    if name_arch.startswith('High') and name_data != 'VOC':
        model_dict = model.state_dict()
        checkpoint = {
            k[6:]: v
            for k, v in checkpoint.items() if k[6:] in model_dict.keys()
        }
        model.load_state_dict(checkpoint)
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(checkpoint)
    print('model loaded!')
    model.eval()
    cnt = 0
    atk_m0 = METRICS(num_classes)
    atk_m1 = METRICS(num_classes)
    RUN_NUM = 500
    loss = nn.CrossEntropyLoss(ignore_index=config['ignore_index'],
                               reduction='none')
    
    steps = 50

    for batch_idx, (data, targ) in enumerate(val_loader):
        cnt += 1
        torch.cuda.empty_cache()
        data = data.float()
        data = data.cuda().requires_grad_()

        targ = targ.cuda()
        output = model(data)

        los = loss(output, targ)
        #print(output.shape)
        save_image(data[0], './input_images/input.jpg')
        pixAcc0, mIoU0, class_IOU0 , mACC0 = atk_m0.update_metrics(output, targ, num_classes)
        #print(pixAcc0, mIoU0, pixAccForOne)
        
        # transform = T.ToPILImage()
        custom_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (128, 0, 0),      # Maroon
            (0, 128, 0),      # Green (dark)
            (0, 0, 128),      # Navy
            (128, 128, 0),    # Olive
            (128, 0, 128),    # Purple
            (0, 128, 128),    # Teal
            (192, 192, 192),  # Silver
            (128, 128, 128),  # Gray
            (255, 165, 0),    # Orange
            (255, 192, 203),  # Pink
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
            (255, 99, 71),    # Tomato
            (255, 140, 0),    # DarkOrange
            (0, 128, 0)       # Green (dark)
        ]

        save_image(data[0], './input_images/input ' + str(cnt) + '.jpg')
        save_target(targ, './labels', 'label '+str(cnt), custom_palette)
        save_images(output, targ, './clean_predictions', 'clean_pred '+str(cnt), custom_palette )
        

        adv = get_adv_examples_CRPGD(data,
                                     targ,
                                     model,
                                     loss,
                                     args.norm,
                                     args.eps,
                                     2.5 * args.eps / steps,
                                     steps,
                                     N=args.n,
                                     SIGMA=args.sigma,
                                     BATCH=args.batch,
                                     a=args.a,
                                     b=args.b,
                                     INTEV=args.interval)
        output = model(adv)
        pixAcc1, mIoU1, class_IOU1, mACC1 = atk_m1.update_metrics(output, targ, num_classes)
        
        save_images(output, targ, './adversarial_predictions', 'adv_pred '+str(cnt), custom_palette )
        
        if cnt % 1 == 0:
            with open(args.log, 'a') as f:
                print("ROUND %d" % (cnt), file=f)
                print("Clean %f %f %f" % (pixAcc0, mIoU0, mACC0), file=f)
                print("OUR  %f %f %f" % (pixAcc1, mIoU1, mACC1), file=f)

                print("ROUND %d" % (cnt))
                print("Clean %f %f %f" % (pixAcc0, mIoU0, mACC0))
                print(len(class_IOU0) )
                print("OUR  %f %f %f" % (pixAcc1, mIoU1, mACC1))
                # print(class_IOU1)

                f.close()

        if cnt == RUN_NUM:
            break


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config',
                        default='configvoc_psp.json',
                        type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--device',
                        default=None,
                        type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--norm', default='2', type=str, help='attack norm')
    parser.add_argument('--sigma', default=0.001, type=float, help='sigma')
    parser.add_argument('--eps', default=1, type=float, help='eps')
    parser.add_argument('--a', default=2, type=int, help='A')
    parser.add_argument('--b', default=4, type=int, help='B')
    parser.add_argument('--n', default=8, type=int, help='N')
    parser.add_argument('--batch',
                        default=8,
                        type=int,
                        help='batch size for RS')
    parser.add_argument('--interval', default=5, type=int, help='interval')
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args, config)
