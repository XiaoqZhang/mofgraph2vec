import sklearn

def get_numpy_regression_metrics(target, pred, prefix: str = ""):
    """ Supervised regression metrics. """
    metrics = {}
    metrics[f"{prefix}_mse"] = sklearn.metrics.mean_squared_error(pred, target)
    metrics[f"{prefix}_mae"] = sklearn.metrics.mean_absolute_error(pred, target)
    metrics[f"{prefix}_r2"] = sklearn.metrics.r2_score(target, pred)
    return metrics