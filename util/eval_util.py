import sklearn.metrics as sk_metrics
import numpy as np

def unravel(probs, gt):
    """Converts probs, preds to flat numpy arrays
       and returns gt"""

    probs = np.concatenate(probs).ravel().tolist()
    probs = np.array(probs)

    gt = np.concatenate(gt).ravel().tolist()
    gt = np.array(gt)
    gt = gt.astype(int)

    return probs, gt

class AverageMeter(object):
    """Computes and stores the average and current value.
    Adapted from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_metric_fn(metric_name):

    # Functions that take probs as input
    fn_dict_probs = {
        'AUPRC': sk_metrics.average_precision_score,
        'AUROC': sk_metrics.roc_auc_score,
        'log_loss': sk_metrics.log_loss,
        'PRC': sk_metrics.precision_recall_curve,
        'ROC': sk_metrics.roc_curve,
        'MSE': sk_metrics.mean_squared_error,
    }

    # Functions that take binary as input
    fn_dict_binary = {
        'accuracy': sk_metrics.accuracy_score,
        'precision': sk_metrics.precision_score,
        'recall': sk_metrics.recall_score,
        'jaccard_similarity': sk_metrics.jaccard_similarity_score,
        'f1': sk_metrics.f1_score
    }

    if metric_name in fn_dict_probs:
        return fn_dict_probs[metric_name], 'probs'
    elif metric_name in fn_dict_binary:
        return fn_dict_binary[metric_name], 'binary'
    else:
        raise ValueError(f'{metric_name} is not in the list of metric evaluation functions')


