<h1 align="center"> Uncertainty Quantification via the Predictability, Computability, Stability (PCS) Framework </h1>

<p align="center">  PCS UQ is a Python library for generating prediction intervals/sets via the PCS framework. Experiments in our paper show that PCS UQ reduces average prediction intervals significantly compared to leading conformal inference methods. 

</p>

## Set-up 

### Installation 

```bash
clone then pip install -e .
```


### Environment Setup 

Set up the environment with the following commands: 
```bash
conda create -n pcs_uq python=3.10 
pip install -r requirements.txt 
pip install -e . 
```


## Usage

The authors of CLEAR have modified the `README.md` to provide a simple example of how to use PCS UQ to generate prediction intervals/sets. 
```python
from PCS.regression import PCS_UQ, PCS_OOB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
# generate synthetic data
X = np.random.randn(100, 10)
y = np.random.randn(100)
estimators = {"rf": RandomForestRegressor(), "lr": LinearRegression()}
pcs_uq = PCS_UQ(num_bootstraps = 100, models = estimators) # initialize the PCS object and provide list of models to fit as well as number of bootstraps
pcs_oob = PCS_OOB(num_bootstraps = 100, models = estimators) # initialize the PCS object and provide list of models to fit as well as number of bootstraps
pcs_uq.fit(X, y) # fit the model
pcs_uq.calibrate(X,y) # calibrate the model
pcs_uq.predict(X) # generate prediction intervals/sets
pcs_oob.fit(X, y) # fit the model
pcs_oob.calibrate(X,y) # calibrate the model
pcs_oob.predict(X) # generate prediction intervals/sets
```

More complete example for usage within CLEAR:
```python
from PCS.regression import PCS_UQ, PCS_OOB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
# generate synthetic data
X = np.random.randn(100, 10)
y = np.random.randn(100)
#--------------------------------
## Point estimators (original PCS)
# point_estimators = {"rf": RandomForestRegressor(), "lr": LinearRegression()}
#--------------------------------
## Quantile estimators (new in CLEAR)
from quantile_forest import RandomForestQuantileRegressor
from xgboost import XGBRegressor
from pygam import ExpectileGAM
quantile_estimators = {
    "rf": RandomForestQuantileRegressor(random_state=777, n_estimators=100, default_quantiles=0.5), 
    "xgb": XGBRegressor(random_state=777, objective='reg:quantileerror', n_estimators=100, tree_method='hist', quantile_alpha=0.5),
    "egam": ExpectileGAM(expectile=0.5, n_splines=10, lam=0.01)
    }

## UQ fit, calibration, and interval predictions (calibrated and raw respectively)
# initialize the PCS object and provide list of models to fit as well as number of bootstraps, top number of models to choose and the alpha
pcs_uq = PCS_UQ(num_bootstraps = 100, models = quantile_estimators, top_k = 1, alpha = 0.05)
pcs_uq.fit(X, y)
pcs_uq.calibrate(X,y)
pcs_uq.predict(X)
pcs_uq.get_intervals(X)

# OOB fit, calibration, and calibrated and interval predictions (calibrated and raw respectively
# Same procedure for OOB
pcs_oob = PCS_OOB(num_bootstraps = 100, models = quantile_estimators, top_k = 1, alpha = 0.05)
pcs_oob.fit(X, y)
pcs_oob.calibrate(X,y)
pcs_oob.predict(X)
pcs_oob.get_intervals(X)
```