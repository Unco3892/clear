import unittest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from sklearn.exceptions import NotFittedError

from src.clear.clear import CLEAR

class TestCLEAR(unittest.TestCase):

    def setUp(self):
        """Set up common test data and a CLEAR model instance."""
        self.random_state = 42
        self.n_samples = 100
        self.n_features = 3
        self.n_bootstraps_test = 5 # Keep low for speed

        # Synthetic data
        self.X_train = np.random.rand(self.n_samples, self.n_features)
        self.y_train = np.sum(self.X_train, axis=1) + np.random.normal(0, 1, self.n_samples)
        self.X_calib = np.random.rand(self.n_samples // 2, self.n_features)
        self.y_calib = np.sum(self.X_calib, axis=1) + np.random.normal(0, 1, self.n_samples // 2)
        self.X_test = np.random.rand(self.n_samples // 2, self.n_features)

        self.clear_model = CLEAR(
            desired_coverage=0.9,
            n_bootstraps=self.n_bootstraps_test,
            random_state=self.random_state,
            n_jobs=1 # For easier debugging and consistency in tests
        )

        # Mocked epistemic and aleatoric predictions for calibration/prediction
        self.median_epistemic_calib = np.random.rand(len(self.y_calib))
        self.lower_epistemic_calib = self.median_epistemic_calib - np.random.rand(len(self.y_calib)) * 0.2
        self.upper_epistemic_calib = self.median_epistemic_calib + np.random.rand(len(self.y_calib)) * 0.2

        self.median_aleatoric_calib = np.random.rand(len(self.y_calib))
        self.lower_aleatoric_calib = self.median_aleatoric_calib - np.random.rand(len(self.y_calib)) * 0.1
        self.upper_aleatoric_calib = self.median_aleatoric_calib + np.random.rand(len(self.y_calib)) * 0.1

        self.median_epistemic_test = np.random.rand(len(self.X_test))
        self.lower_epistemic_test = self.median_epistemic_test - np.random.rand(len(self.X_test)) * 0.2
        self.upper_epistemic_test = self.median_epistemic_test + np.random.rand(len(self.X_test)) * 0.2

        self.median_aleatoric_test = np.random.rand(len(self.X_test))
        self.lower_aleatoric_test = self.median_aleatoric_test - np.random.rand(len(self.X_test)) * 0.1
        self.upper_aleatoric_test = self.median_aleatoric_test + np.random.rand(len(self.X_test)) * 0.1

        # Initial state for fit_on_residuals
        self.clear_model.fit_on_residuals = False
        # Mock epistemic models for tests that need them pre-fitted for aleatoric part
        self.clear_model.epistemic_models = []
        for _ in range(self.n_bootstraps_test):
            model_mock = MagicMock()
            # model_mock.predict.return_value = np.random.rand(self.n_samples)
            # Make predict a MagicMock that returns based on input length
            model_mock.predict = MagicMock(side_effect=lambda x_input: np.random.rand(len(x_input)))
            # Make it appear fitted for predict_epistemic calls if check_is_fitted is used there too
            model_mock.__sklearn_is_fitted__ = lambda: True 
            model_mock.fitted_ = True
            self.clear_model.epistemic_models.append(model_mock)

    def test_initialization(self):
        self.assertEqual(self.clear_model.desired_coverage, 0.9)
        self.assertEqual(self.clear_model.n_bootstraps, self.n_bootstraps_test)
        self.assertIsNotNone(self.clear_model.lambdas)

    @patch('src.clear.clear.CLEAR._fit_single_bootstrap')
    def test_fit_aleatoric_standard(self, mock_fit_single_bootstrap):
        # Mock the internal fitting of bootstrap models
        # Each call to _fit_single_bootstrap should return 3 mocked models
        mock_model = MagicMock()
        mock_model.fit.return_value = None # Ensure fit can be called
        mock_model.__sklearn_is_fitted__ = lambda: True # Make it appear fitted
        mock_model.fitted_ = True 

        mock_fit_single_bootstrap.return_value = (mock_model, mock_model, mock_model)

        self.clear_model.fit_aleatoric(self.X_train, self.y_train, quantile_model='rf', model_params={'n_estimators': 5})
        self.assertEqual(len(self.clear_model.lower_models), self.n_bootstraps_test)
        self.assertEqual(len(self.clear_model.median_models), self.n_bootstraps_test)
        self.assertEqual(len(self.clear_model.upper_models), self.n_bootstraps_test)
        self.assertFalse(self.clear_model.fit_on_residuals)
        self.assertEqual(mock_fit_single_bootstrap.call_count, self.n_bootstraps_test)

    @patch('src.clear.clear.CLEAR._fit_single_bootstrap')
    def test_fit_aleatoric_residual(self, mock_fit_single_bootstrap):
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.__sklearn_is_fitted__ = lambda: True # Make it appear fitted
        mock_model.fitted_ = True

        mock_fit_single_bootstrap.return_value = (mock_model, mock_model, mock_model)

        # Mock epistemic predictions for residual fitting
        mock_epistemic_preds_train = np.random.rand(len(self.y_train))

        self.clear_model.fit_aleatoric(
            self.X_train, self.y_train,
            quantile_model='rf',
            model_params={'n_estimators': 5},
            fit_on_residuals=True,
            epistemic_preds=mock_epistemic_preds_train
        )
        self.assertTrue(self.clear_model.fit_on_residuals)
        self.assertEqual(mock_fit_single_bootstrap.call_count, self.n_bootstraps_test)
        # Check that median_preds was passed to _fit_single_bootstrap
        # args[5] is median_preds in _fit_single_bootstrap(X, y, model, params, i, median_preds)
        # Iterate through all calls to check if median_preds was passed correctly
        for call_args_tuple in mock_fit_single_bootstrap.call_args_list:
            passed_median_preds = call_args_tuple[0][5] # call_args_tuple[0] is the tuple of positional args
            self.assertIsNotNone(passed_median_preds, "median_preds should be passed when fit_on_residuals=True")
            np.testing.assert_array_equal(passed_median_preds, mock_epistemic_preds_train, 
                                          "median_preds passed to _fit_single_bootstrap does not match mock epistemic preds")

    def test_predict_aleatoric_standard(self):
        # Prepare mock models that appear fitted
        fitted_mock_models = []
        for _ in range(self.n_bootstraps_test):
            m = MagicMock(predict=MagicMock(return_value=np.random.rand(len(self.X_test))))
            m.__sklearn_is_fitted__ = lambda: True
            m.fitted_ = True 
            fitted_mock_models.append(m)

        self.clear_model.lower_models = list(fitted_mock_models) 
        self.clear_model.median_models = list(fitted_mock_models)
        self.clear_model.upper_models = list(fitted_mock_models)
        self.clear_model.fit_on_residuals = False

        median_pred, lower_pred, upper_pred = self.clear_model.predict_aleatoric(self.X_test)
        self.assertEqual(median_pred.shape, (len(self.X_test),))
        self.assertEqual(lower_pred.shape, (len(self.X_test),))
        self.assertEqual(upper_pred.shape, (len(self.X_test),))
        self.assertTrue(np.all(lower_pred <= median_pred) and np.all(median_pred <= upper_pred))

    def test_predict_aleatoric_residual(self):
        # Prepare mock models that appear fitted
        fitted_mock_models = []
        for _ in range(self.n_bootstraps_test):
            m = MagicMock(predict=MagicMock(return_value=np.random.rand(len(self.X_test))))
            m.__sklearn_is_fitted__ = lambda: True
            m.fitted_ = True 
            fitted_mock_models.append(m)
            
        self.clear_model.lower_models = list(fitted_mock_models)
        self.clear_model.median_models = list(fitted_mock_models)
        self.clear_model.upper_models = list(fitted_mock_models)
        self.clear_model.fit_on_residuals = True

        mock_epistemic_preds_test = np.random.rand(len(self.X_test))
        # setUp already configures self.clear_model.epistemic_models to be "fitted"
        
        median_pred, lower_pred, upper_pred = self.clear_model.predict_aleatoric(self.X_test, epistemic_preds=mock_epistemic_preds_test)
        self.assertEqual(median_pred.shape, (len(self.X_test),))
        self.assertEqual(lower_pred.shape, (len(self.X_test),))
        self.assertEqual(upper_pred.shape, (len(self.X_test),))
        # Check that predict was called on augmented X
        for model_mock in self.clear_model.lower_models:
            args_call, _ = model_mock.predict.call_args
            self.assertEqual(args_call[0].shape[1], self.n_features + 1)

    @patch('src.clear.clear.LinearGAM') 
    def test_fit_epistemic(self, MockLinearGAM):
        mock_gam_instance = MockLinearGAM.return_value
        mock_gam_instance.fit.return_value = mock_gam_instance 
        mock_gam_instance.__sklearn_is_fitted__ = lambda: True 
        mock_gam_instance.fitted_ = True

        self.clear_model.fit_epistemic(self.X_train, self.y_train)
        self.assertEqual(len(self.clear_model.epistemic_models), self.n_bootstraps_test)
        self.assertEqual(MockLinearGAM.call_count, self.n_bootstraps_test)

    @patch('src.clear.clear.compute_quantile_ensemble_bounds') 
    def test_predict_epistemic(self, mock_compute_bounds):
        # Setup mock return value for compute_quantile_ensemble_bounds
        mock_median = np.random.rand(len(self.X_test))
        mock_lower = mock_median - 0.1
        mock_upper = mock_median + 0.1
        mock_compute_bounds.return_value = (mock_median, mock_lower, mock_upper)

        # epistemic_models are set up as "fitted" in self.setUp
        
        median_pred, lower_pred, upper_pred, ensemble_preds = self.clear_model.predict_epistemic(self.X_test)

        self.assertEqual(median_pred.shape, (len(self.X_test),))
        self.assertEqual(lower_pred.shape, (len(self.X_test),))
        self.assertEqual(upper_pred.shape, (len(self.X_test),))
        self.assertEqual(ensemble_preds.shape, (self.n_bootstraps_test, len(self.X_test)))
        mock_compute_bounds.assert_called_once()

    def test_calibrate_and_predict(self):
        # This is more of an integration test for calibrate and predict
        # We need to have some aleatoric and epistemic predictions available
        # For simplicity, we'll mock the outputs of predict_aleatoric and predict_epistemic
        # or rather, directly set the necessary attributes as if they were predicted.

        self.clear_model.calibrate(
            self.y_calib,
            median_epistemic=self.median_epistemic_calib,
            aleatoric_median=self.median_aleatoric_calib,
            aleatoric_lower=self.lower_aleatoric_calib,
            aleatoric_upper=self.upper_aleatoric_calib,
            epistemic_lower=self.lower_epistemic_calib,
            epistemic_upper=self.upper_epistemic_calib,
            verbose=False
        )
        self.assertIsNotNone(self.clear_model.gamma)
        self.assertIsNotNone(self.clear_model.optimal_lambda)

        # Mock the internal predict_epistemic and predict_aleatoric calls within clear_model.predict
        # by providing external predictions
        lower_bounds, upper_bounds = self.clear_model.predict(
            self.X_test,
            external_epistemic={
                'median': self.median_epistemic_test,
                'lower': self.lower_epistemic_test,
                'upper': self.upper_epistemic_test
            },
            external_aleatoric={
                'median': self.median_aleatoric_test,
                'lower': self.lower_aleatoric_test,
                'upper': self.upper_aleatoric_test
            }
        )
        self.assertEqual(lower_bounds.shape, (len(self.X_test),))
        self.assertEqual(upper_bounds.shape, (len(self.X_test),))
        self.assertTrue(np.all(lower_bounds <= upper_bounds))

    def test_calibrate_fixed_gamma_lambda(self):
        fixed_gamma = 1.0
        fixed_lambda = 0.5
        clear_model_fixed = CLEAR(desired_coverage=0.9, fixed_gamma=fixed_gamma, fixed_lambda=fixed_lambda)
        clear_model_fixed.calibrate(
            self.y_calib,
            self.median_epistemic_calib, self.median_aleatoric_calib, self.lower_aleatoric_calib, self.upper_aleatoric_calib,
            self.lower_epistemic_calib, self.upper_epistemic_calib, verbose=False
        )
        self.assertEqual(clear_model_fixed.gamma, fixed_gamma)
        self.assertEqual(clear_model_fixed.optimal_lambda, fixed_lambda)

    def test_calibrate_fixed_gamma_optimize_lambda(self):
        fixed_gamma = 1.0
        clear_model_fixed_g = CLEAR(desired_coverage=0.9, fixed_gamma=fixed_gamma)
        clear_model_fixed_g.calibrate(
             self.y_calib,
            self.median_epistemic_calib, self.median_aleatoric_calib, self.lower_aleatoric_calib, self.upper_aleatoric_calib,
            self.lower_epistemic_calib, self.upper_epistemic_calib, verbose=False
        )
        self.assertEqual(clear_model_fixed_g.gamma, fixed_gamma)
        self.assertIsNotNone(clear_model_fixed_g.optimal_lambda) # Lambda should be optimized

    def test_calibrate_fixed_lambda_optimize_gamma(self):
        fixed_lambda = 0.5
        clear_model_fixed_l = CLEAR(desired_coverage=0.9, fixed_lambda=fixed_lambda)
        clear_model_fixed_l.calibrate(
            self.y_calib,
            self.median_epistemic_calib, self.median_aleatoric_calib, self.lower_aleatoric_calib, self.upper_aleatoric_calib,
            self.lower_epistemic_calib, self.upper_epistemic_calib, verbose=False
        )
        self.assertEqual(clear_model_fixed_l.optimal_lambda, fixed_lambda)
        self.assertIsNotNone(clear_model_fixed_l.gamma) # Gamma should be optimized

    def test_predict_uncalibrated_raises_error(self):
        self.clear_model.gamma = None # Ensure not calibrated
        with self.assertRaises(ValueError):
            self.clear_model.predict(self.X_test)

if __name__ == '__main__':
    unittest.main() 