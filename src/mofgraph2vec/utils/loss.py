import torchmetrics
import sklearn

def get_regression_metrics(target, pred, prefix: str = ""):
    """Get regression metrics."""
    metrics = {}
    metrics[f"{prefix}_mse"] = torchmetrics.functional.mean_squared_error(pred, target)
    metrics[f"{prefix}_mae"] = torchmetrics.functional.mean_absolute_error(pred, target)
    metrics[f"{prefix}_r2"] = torchmetrics.functional.r2_score(pred, target)
    return metrics

def get_numpy_regression_metrics(target, pred, prefix: str = ""):
    metrics = {}
    metrics[f"{prefix}_mse"] = sklearn.metrics.mean_squared_error(pred, target)
    metrics[f"{prefix}_mae"] = sklearn.metrics.mean_absolute_error(pred, target)
    metrics[f"{prefix}_r2"] = sklearn.metrics.r2_score(pred, target)
    return metrics