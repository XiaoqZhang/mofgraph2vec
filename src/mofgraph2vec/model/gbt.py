from mofgraph2vec.utils.loss import get_numpy_regression_metrics
from xgboost import XGBRegressor

class Regressor(XGBRegressor):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()
        self.metrics = get_numpy_regression_metrics
    
    def test(self, x, y, target_transform):
        pred = target_transform.inverse_transform(self.predict(x).reshape(-1,1))
        y = target_transform.inverse_transform(y)
        loss = self.metrics(pred, y, "test")
        return loss