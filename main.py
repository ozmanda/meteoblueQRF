from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets

x, y = datasets.fetch_california_housing(return_X_y=True)

qrf = RandomForestQuantileRegressor()
qrf.fit(x, y)

yPred = qrf.predict(x, quantiles=[0.025, 0.5, 0.975])