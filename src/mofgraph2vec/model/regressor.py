from mofgraph2vec.utils.loss import get_numpy_regression_metrics
from xgboost import XGBRegressor

class Regressor(XGBRegressor):
    def __init__(
        self,
        n_estimators,
        max_depth,
        reg_alpha,
        reg_lambda,
        subsample,
        min_child_weight,
        learning_rate,
        random_state,
        **kwargs
    ):
        super().__init__(
            n_estimators = n_estimators,
            max_depth = max_depth,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            subsample = subsample,
            min_child_weight = min_child_weight,
            random_state = random_state,
            learning_rate = learning_rate,
        )
        self.metrics = get_numpy_regression_metrics
    
    def test(self, x, y, target_transform):
        pred = target_transform.inverse_transform(self.predict(x).reshape(-1,1))
        y = target_transform.inverse_transform(y)
        loss = self.metrics(pred, y, "test")
        return loss