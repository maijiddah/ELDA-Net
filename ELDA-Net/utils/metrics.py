import numpy as np

def compute_metrics(pred, label):
    tp = np.logical_and(pred, label).sum()
    fp = np.logical_and(pred, np.logical_not(label)).sum()
    fn = np.logical_and(np.logical_not(pred), label).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return f1, iou
