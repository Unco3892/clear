# Uncertainty Aware Conformalized Quantile Regression (UACQR)
Here we provide an implementation of UACQR and notebooks to reproduce the figures and tables from ["Integrating Uncertainty Awareness into Conformalized Quantile Regression"](https://arxiv.org/abs/2306.08693)

We demonstrate how to use our method for neural networks and Quantile Regression Forests in the two demo notebooks.  

For more information on the hyperparameter options for Quantile Regression Forests, see sklearn_quantile documentation: https://sklearn-quantile.readthedocs.io/en/latest/

For more information on the hyperparameter options and architectural details for our neural network implementation, see helper.py

For real data experiments, the GetDatasets module and associated assets are a fork from https://github.com/msesia/cqr-comparison

## Note from CLEAR authors
The original repository contained some mistakes that we corrected, and added parallelization for the experiments. Additionally, it was unclear whether the experiments were ran with standardized outputs, but given the difference in the scale of the measurements, we strongly believe that it is the case. We have also removed `sklearn_quantile` due to conflicts with the other packages, and use `quantile-forest` instead to also be more in line with the CLEAR paper.
<!-- We have adapted `demo_nn.py` and `demo_rf.py` from the original UACQR notebooks. -->
