import torch
import numpy as np
from loss import metric, compute_iou_batch
from predict import predict

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice= metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice, iou = meter.get_metrics()
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f" % (epoch_loss, iou, dice))
    return dice, iou