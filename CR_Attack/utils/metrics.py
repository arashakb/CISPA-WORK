import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target, num_class, labeled):
    pixel_labeled = labeled.sum()
    obj_correct = []
    obj_total = [] 
    #print(pixel_labeled)
    
    pixel_correct = ((predict == target) * labeled).sum()
    for i in range(num_class):
        sum_of_object = ((predict == target) * (predict == i) * labeled).sum()
        number_labeled_object = ((target == i) * labeled).sum()
        #print(sum_of_object.cpu().numpy(), number_labeled_object.cpu().numpy())
        obj_correct.append(sum_of_object)
        obj_total.append(number_labeled_object) 
    
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy(), torch.tensor(obj_correct).cpu().numpy(), torch.tensor(obj_total).cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):

    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()
    
    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled, each_object_correct, each_object_total_number = batch_pix_accuracy(predict, target, num_class, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(each_object_correct, 5), np.round(each_object_total_number, 5)]
