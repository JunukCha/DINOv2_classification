import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_acc(y_pred, y_true, threshold=0.7):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    max_probs = np.max(y_pred, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    correct = ((y_pred_labels == y_true) & (max_probs >= threshold)).sum()
    incorrect = (max_probs < threshold).sum()
    total = correct + incorrect

    accuracy = correct / total
    return accuracy

def cal_acc_org(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = correct / total
    return accuracy