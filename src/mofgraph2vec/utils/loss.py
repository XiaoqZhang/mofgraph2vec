import torchmetrics

def get_regression_metrics(target, pred, prefix: str = ""):
    """Get regression metrics."""
    metrics = {}
    metrics[f"{prefix}_mse"] = torchmetrics.functional.mean_squared_error(pred, target)
    metrics[f"{prefix}_mae"] = torchmetrics.functional.mean_absolute_error(pred, target)
    metrics[f"{prefix}_r2"] = torchmetrics.functional.r2_score(pred, target)
    return metrics