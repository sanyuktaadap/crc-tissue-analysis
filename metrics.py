import torchmetrics as tm

def get_metrics(y_hat, y, metric=None):
    if metric == "accuracy":
        fn = tm.Accuracy(task="multiclass", num_classes=4)
        score = fn(y_hat, y)
    elif metric == "specificity":
        fn = tm.classification.MulticlassSpecificity(num_classes=4, average=None)
        score = fn(y_hat, y)
    elif metric == "precision":
        fn = tm.classification.MulticlassPrecision(num_classes=4, average=None)
        score = fn(y_hat, y)
    elif metric == "recall":
        fn = tm.classification.MulticlassRecall(num_classes=4, average=None)
        score = fn(y_hat, y)
    elif metric == "f1":
        fn = tm.classification.MulticlassF1Score(num_classes=4, average=None)
        score = fn(y_hat, y)

    return score