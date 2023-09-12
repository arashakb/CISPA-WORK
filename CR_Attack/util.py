import numpy as np
import random
import torch
from utils.helpers import colorize_mask
import torch
import os
import matplotlib.pyplot as plt
from utils.metrics import eval_metrics
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    

class METRICS:

    def __init__(self, num_classes):
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0
        self.num_classes = num_classes
        self.total_each_object_correct = [0] * self.num_classes
        self.total_each_object_total_number = [0] * self.num_classes

    def update_metrics(self, output, targ, num_classes):
        seg = eval_metrics(output, targ, num_classes)
        self.update_seg_metrics(*seg)
        return self.get_seg_metrics().values()

    def update_seg_metrics(self, correct, labeled, inter, union, each_object_correct, each_object_total_number):

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_each_object_correct += each_object_correct 
        self.total_each_object_total_number += each_object_total_number

    def get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        object_arr = []
        for i in range(0, self.num_classes):
            if self.total_each_object_total_number[i] != 0:
                object_arr.append(1.0 * self.total_each_object_correct[i] / (np.spacing(1) + self.total_each_object_total_number[i]))
        mACC = sum(object_arr) / len(object_arr)
        IoU = []
        for i in range(0, self.num_classes):
            if (self.total_union[i] > 0):
                IoU.append(self.total_inter[i] / self.total_union[i])
        mIoU = sum(IoU) / len(IoU)
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "Pixel_Accuracy_for_obj_1": np.round(mACC, 3)
        }


def save_target(targ, output_path, name, palette):
    target = targ.clone()
    target = target.detach().cpu().squeeze(0)
    target[target == 255] = 0
    y = target.numpy()
    # colorized_mask = colorize_mask(y, palette)
    
    colorized_mask = Image.fromarray(y.astype('uint8'), mode='P')
    colorized_mask.putpalette([c for color in palette for c in color])

    colorized_mask.save(os.path.join(output_path, name + '.png'))


def save_images(output, targ, output_path, name, palette):
    # Saves the image, the model output and the results after the post processing
    mask = output.detach().cpu()
    target = targ.clone()
    target[target == 255] = 0

    y = torch.argmax(mask, dim=1).squeeze(0)
    y = y.numpy()
    # colorized_mask = colorize_mask(y, palette)

    colorized_mask = Image.fromarray(y.astype('uint8'), mode='P')
    colorized_mask.putpalette([c for color in palette for c in color])
    
    colorized_mask.save(os.path.join(output_path, name + '.png'))


def save2d(o, output_path, name):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    o = abs(o[0])
    o = (o - o.min()) / (o.max() - o.min())
    plt.imsave(os.path.join(output_path, '{}').format(name),
               o,
               format='png',
               cmap='hot')
