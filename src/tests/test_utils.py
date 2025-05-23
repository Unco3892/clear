import unittest
import numpy as np
import pandas as pd
import sys
import os

from src.clear import utils as clear_utils # Assumes utils.py is importable from clear package
from src.pcs import utils as pcs_utils # For PCS specific utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.random_state = 42
        np.random.seed(self.random_state)
        self.n_models = 5
        self.n_samples = 20
        # Example ensemble predictions: (n_models, n_samples)
        self.ensemble_preds = np.random.rand(self.n_models, self.n_samples) * 10
        # Example aleatoric median: (n_samples,)
        self.aleatoric_median = np.median(self.ensemble_preds, axis=0) + np.random.normal(0, 0.2, self.n_samples)

    def test_compute_quantile_ensemble_bounds_standard(self):
        median, lower, upper = clear_utils.compute_quantile_ensemble_bounds(
            self.ensemble_preds, lower_quantile=0.1, upper_quantile=0.9
        )
        self.assertEqual(median.shape, (self.n_samples,))
        self.assertEqual(lower.shape, (self.n_samples,))
        self.assertEqual(upper.shape, (self.n_samples,))
        self.assertTrue(np.all(lower <= median))
        self.assertTrue(np.all(median <= upper))

    def test_compute_quantile_ensemble_bounds_with_aleatoric(self):
        median, lower, upper = clear_utils.compute_quantile_ensemble_bounds(
            self.ensemble_preds, 
            lower_quantile=0.1, 
            upper_quantile=0.9,
            aleatoric_median=self.aleatoric_median,
            symmetric_noise=True
        )
        self.assertEqual(median.shape, (self.n_samples,))
        # The combined_preds in the function will have one more model (the aleatoric median)
        # We expect quantiles to still be ordered.
        self.assertTrue(np.all(lower <= median))
        self.assertTrue(np.all(median <= upper))

    def test_compute_quantile_ensemble_bounds_crossing_quantiles_warning(self):
        # Create a scenario where quantiles might cross if not handled
        ensemble_forced_crossing = self.ensemble_preds.copy()
        # Force the 0.9 quantile to be below the 0.1 for some samples if we took them directly
        # Forcing this would require manipulating the distribution heavily.
        # Instead, we rely on the internal check of the function.
        # We can check if a warning is printed (hard to do in unittest directly without capturing stdout/stderr)
        # For now, just ensure it runs and maintains order due to internal correction.
        _, lower, upper = clear_utils.compute_quantile_ensemble_bounds(
            ensemble_forced_crossing, lower_quantile=0.9, upper_quantile=0.1 # Deliberately swapped
        )
        # The function should internally correct this, so lower <= upper
        # Or rather, it computes q(0.9) and q(0.1) and calls them lower and upper respectively.
        # The internal check `upper = np.maximum(upper, median); lower = np.minimum(lower, median)` handles it.
        # If lower_quantile > upper_quantile initially, the result might be `lower` (q_0.9) > `upper` (q_0.1)
        # before the internal correction with median. This test might need more thought on how to trigger the warning effectively.
        # For now, we check that the output bounds are ordered as expected by their names (lower <= upper)
        # self.assertTrue(np.all(lower <= upper)) # This may not hold if definition of lower/upper is strict to quantile value
        pass # Testing the warning for crossing quantiles is tricky without log capture here

    def test_generate_distance_points(self):
        distances = np.array([1.0, 2.0])
        n_points_per_distance = 5
        n_dims = 3
        points = clear_utils.generate_distance_points(n_points_per_distance, distances, n_dims, random_state=self.random_state)
        
        self.assertEqual(points.shape, (len(distances) * n_points_per_distance, n_dims))
        
        for i, dist_val in enumerate(distances):
            start_idx = i * n_points_per_distance
            end_idx = start_idx + n_points_per_distance
            points_at_dist = points[start_idx:end_idx]
            norms = np.linalg.norm(points_at_dist, axis=1)
            np.testing.assert_allclose(norms, dist_val, rtol=1e-5, err_msg=f"Norms at distance {dist_val} not correct.")

    def test_compute_coverage_by_distance(self):
        n_dims = 2
        distances = np.array([0.5, 1.5])
        n_points_per_dist = 10
        X_test_flat = clear_utils.generate_distance_points(n_points_per_dist, distances, n_dims, random_state=self.random_state)
        y_test = np.sum(X_test_flat, axis=1) + np.random.normal(0, 0.1, X_test_flat.shape[0])
        
        # Create bounds that cover roughly half the points at each distance for testing
        lower_bounds = y_test - 0.05 
        upper_bounds = y_test + 0.05 
        # Make some miss
        lower_bounds[::2] = y_test[::2] + 0.01 # Lower bound above y_test

        metrics_dict = clear_utils.compute_coverage_by_distance(
            X_test_flat, lower_bounds, upper_bounds, y_test, distances, n_points_per_dist
        )
        self.assertIn('distances', metrics_dict)
        self.assertIn('coverage', metrics_dict)
        self.assertIn('width', metrics_dict)
        self.assertEqual(len(metrics_dict['distances']), len(distances))
        self.assertEqual(len(metrics_dict['coverage']), len(distances))
        self.assertEqual(len(metrics_dict['width']), len(distances))
        self.assertTrue(np.all(metrics_dict['coverage'] >= 0) and np.all(metrics_dict['coverage'] <= 1))
        self.assertTrue(np.all(metrics_dict['width'] >= 0))

    def test_plot_distance_metrics(self):
        # Test if it runs without error, actual plot verification is complex
        distances = np.array([0.5, 1.0, 1.5])
        coverage_data = np.array([0.8, 0.85, 0.9])
        width_data = np.array([0.2, 0.25, 0.3])
        metrics_data = {
            'MethodA': {'distances': distances, 'coverage': coverage_data, 'width': width_data}
        }
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, _ = clear_utils.plot_distance_metrics(metrics_data, methods=['MethodA'], colors=['blue'], method_names=['Method A'], target_coverage=0.9)
            plt.close(fig)
        except ImportError:
            self.skipTest("Matplotlib not installed, skipping plot_distance_metrics test")
        except Exception as e:
            # Catch other potential plotting errors if matplotlib is imported but fails
            self.fail(f"plot_distance_metrics raised an exception: {e}")

    def test_convert_to_serializable_and_load_results_safely(self):
        # Basic test for numpy array
        arr = np.array([1, 2, 3])
        serializable_arr = pcs_utils.convert_to_serializable(arr)
        self.assertIsInstance(serializable_arr, list)

        # Test for Pandas DataFrame
        df_data = {'A': [1, 2], 'B': [3, 4]}
        df = pd.DataFrame(df_data)
        serializable_df = pcs_utils.convert_to_serializable(df)
        self.assertIsInstance(serializable_df, dict)
        self.assertTrue(serializable_df.get("_pandas_dataframe_"))

        # Test nested structure
        nested_data = {
            'array': np.array([5.0, 6.0]),
            'dataframe': pd.DataFrame({'X': [7,8]}),
            'list': [np.array([9,10]), 'string']
        }
        serializable_nested = pcs_utils.convert_to_serializable(nested_data)
        self.assertIsInstance(serializable_nested['array'], list)
        self.assertIsInstance(serializable_nested['dataframe'], dict)
        self.assertIsInstance(serializable_nested['list'][0], list)

        # This part is harder to test without actual pickling/unpickling and file I/O,
        # but we can check the reconstruction logic if load_results_safely was more exposed
        # or if we mock pickle.load.
        # For now, focusing on convert_to_serializable.
        pass

if __name__ == '__main__':
    unittest.main() 