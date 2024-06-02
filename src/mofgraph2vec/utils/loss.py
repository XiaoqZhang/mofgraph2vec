import sklearn

def get_numpy_regression_metrics(target, pred, prefix: str = ""):
    """ Supervised regression metrics. 

    Args:
        target (np.array): the target values
        pred (np.array): the predicted values
        prefix (str, optional): prefix name. Defaults to "".

    Returns:
        _type_: _description_
    """
    metrics = {}
    metrics[f"{prefix}_mse"] = sklearn.metrics.mean_squared_error(pred, target)
    metrics[f"{prefix}_mae"] = sklearn.metrics.mean_absolute_error(pred, target)
    metrics[f"{prefix}_r2"] = sklearn.metrics.r2_score(target, pred)
    return metrics