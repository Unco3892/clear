import unittest
import numpy as np
import sys
import os

from src.clear import metrics

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.lower_perfect = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.upper_perfect = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) # Covers all
        
        self.lower_narrow = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        self.upper_narrow = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]) # Covers all, but narrower

        self.lower_wide = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.upper_wide = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) # Covers all, wider

        self.lower_miss_some = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 100, 100, 100, 100, 100]) 
        self.upper_miss_some = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 101, 101, 101, 101, 101]) # Misses last 5 points
        
        self.alpha_test = 0.1 # For 90% intervals
        self.f_median = (self.lower_narrow + self.upper_narrow) / 2 # Example median predictions

    def test_picp(self):
        self.assertAlmostEqual(metrics.picp(self.y_true, self.lower_perfect, self.upper_perfect), 1.0)
        self.assertAlmostEqual(metrics.picp(self.y_true, self.lower_miss_some, self.upper_miss_some), 0.5) # First 5 covered
        self.assertAlmostEqual(metrics.picp(self.y_true, self.upper_perfect, self.lower_perfect), 0.0) # Flipped bounds

    def test_mpiw(self):
        self.assertAlmostEqual(metrics.mpiw(self.lower_perfect, self.upper_perfect), 2.0)
        self.assertAlmostEqual(metrics.mpiw(self.lower_narrow, self.upper_narrow), 1.0)
        self.assertAlmostEqual(metrics.mpiw(self.lower_wide, self.upper_wide), 4.0)

    def test_niw(self):
        data_range = np.max(self.y_true) - np.min(self.y_true)
        self.assertAlmostEqual(metrics.niw(self.y_true, self.lower_perfect, self.upper_perfect), 2.0 / data_range)
        # Test with zero range in y_true
        y_true_const = np.array([5, 5, 5])
        lower_const = np.array([4, 4, 4])
        upper_const = np.array([6, 6, 6])
        self.assertAlmostEqual(metrics.niw(y_true_const, lower_const, upper_const), metrics.mpiw(lower_const, upper_const))


    def test_quantile_loss(self):
        # Test case: perfect prediction at median (tau=0.5)
        self.assertAlmostEqual(metrics.quantile_loss(self.y_true, self.y_true, 0.5), 0.0)
        # Test with known values
        y = np.array([1, 2, 3])
        q_high = np.array([2, 3, 4]) # Predictions are too high
        q_low = np.array([0, 1, 2])   # Predictions are too low
        tau = 0.1
        # Loss for q_high: (1-0.1)*(2-1) + (1-0.1)*(3-2) + (1-0.1)*(4-3) = 0.9 * 3 = 2.7. Mean = 0.9
        self.assertAlmostEqual(metrics.quantile_loss(y, q_high, tau), np.mean([(1-tau)*(2-1), (1-tau)*(3-2), (1-tau)*(4-3)]))
        # Loss for q_low: 0.1*(1-0) + 0.1*(2-1) + 0.1*(3-2) = 0.1 * 3 = 0.3. Mean = 0.1
        self.assertAlmostEqual(metrics.quantile_loss(y, q_low, tau), np.mean([tau*(1-0), tau*(2-1), tau*(3-2)]))

    def test_compute_quantile_loss(self):
        # If bounds are perfect medians, and alpha implies tau=0.5, loss should be low
        # This is not a typical use, as lower/upper are not medians
        # We expect compute_quantile_loss to use alpha/2 and 1-alpha/2
        avg_ql = metrics.compute_quantile_loss(self.lower_narrow, self.upper_narrow, self.y_true, alpha=self.alpha_test)
        self.assertTrue(avg_ql >= 0)

    def test_expectile_loss(self):
        # Test case: perfect prediction at expectile (tau=0.5, equivalent to mean for symmetric error)
        self.assertAlmostEqual(metrics.expectile_loss(self.y_true, self.y_true, 0.5), 0.0)
        y = np.array([1, 2, 3])
        q = np.array([1.5, 1.5, 1.5]) # predictions
        tau_high = 0.9
        # y <= q: (1-0.9)*(1-1.5)^2 = 0.1 * 0.25 = 0.025
        # y > q: 0.9*(2-1.5)^2 = 0.9*0.25 = 0.225;  0.9*(3-1.5)^2 = 0.9*2.25 = 2.025
        # Mean = (0.025 + 0.225 + 2.025) / 3
        expected_loss_high = np.mean([ (1-tau_high)*(1-1.5)**2, tau_high*(2-1.5)**2, tau_high*(3-1.5)**2 ])
        self.assertAlmostEqual(metrics.expectile_loss(y, q, tau_high), expected_loss_high)
        with self.assertRaises(ValueError):
            metrics.expectile_loss(self.y_true, self.y_true, 0) # tau must be > 0
        with self.assertRaises(ValueError):
            metrics.expectile_loss(self.y_true, self.y_true, 1) # tau must be < 1

    def test_compute_average_expectile_loss(self):
        avg_el = metrics.compute_average_expectile_loss(self.lower_narrow, self.upper_narrow, self.y_true, alpha=self.alpha_test)
        self.assertTrue(avg_el >= 0)
        with self.assertRaises(ValueError):
             metrics.compute_average_expectile_loss(self.lower_narrow, self.upper_narrow, self.y_true, alpha=0)
        with self.assertRaises(ValueError):
             metrics.compute_average_expectile_loss(self.lower_narrow, self.upper_narrow, self.y_true, alpha=2)

    def test_crps_interval(self):
        # For perfect coverage and intervals centered on true values, with width matching error dist.
        # This is a simplified check; exact CRPS value depends on assumed uniform dist.
        crps_val = metrics.crps_interval(self.y_true, self.lower_narrow, self.upper_narrow)
        self.assertTrue(crps_val >= 0)
        # Test degenerate case lower=upper
        self.assertAlmostEqual(metrics.crps_interval(np.array([1]), np.array([1]), np.array([1])), 0.0)
        self.assertAlmostEqual(metrics.crps_interval(np.array([2]), np.array([1]), np.array([1])), 1.0)
        with self.assertRaises(ValueError):
            metrics.crps_interval(self.y_true, self.lower_narrow[:-1], self.upper_narrow)

    def test_interval_score_loss(self):
        isl = metrics.interval_score_loss(self.upper_narrow, self.lower_narrow, self.y_true, self.alpha_test)
        self.assertEqual(len(isl), len(self.y_true))
        self.assertTrue(np.all(isl >= 0))
        # Example from UACQR paper for y_true < lower:
        # (upper-lower) + (2/alpha)*(lower-y_true)
        # y_true=1, lower=2, upper=3, alpha=0.1 -> (3-2) + (2/0.1)*(2-1) = 1 + 20*1 = 21
        self.assertAlmostEqual(metrics.interval_score_loss(np.array([3]), np.array([2]), np.array([1]), 0.1)[0], 21)
         # Example for y_true > upper:
        # (upper-lower) + (2/alpha)*(y_true-upper)
        # y_true=4, lower=2, upper=3, alpha=0.1 -> (3-2) + (2/0.1)*(4-3) = 1 + 20*1 = 21
        self.assertAlmostEqual(metrics.interval_score_loss(np.array([3]), np.array([2]), np.array([4]), 0.1)[0], 21)
        # Example for lower <= y_true <= upper:
        # (upper-lower)
        # y_true=2.5, lower=2, upper=3, alpha=0.1 -> (3-2) = 1
        self.assertAlmostEqual(metrics.interval_score_loss(np.array([3]), np.array([2]), np.array([2.5]), 0.1)[0], 1)

    def test_average_interval_score_loss(self):
        aisl = metrics.average_interval_score_loss(self.upper_narrow, self.lower_narrow, self.y_true, self.alpha_test)
        self.assertTrue(aisl >= 0)

    def test_compute_auc_and_nciw(self):
        # These are more complex, testing for non-error and plausible output ranges
        auc_val, auc_info = metrics.compute_auc(self.y_true, self.lower_narrow, self.upper_narrow, f=self.f_median)
        self.assertTrue(0 <= auc_val <= metrics.niw(self.y_true, self.lower_narrow, self.upper_narrow)) # AUC is area under PICP(0-1) vs NIW
        self.assertIn('sorted_c', auc_info)
        self.assertIn('PICPs', auc_info)
        self.assertIn('NMPIWs', auc_info)

        nciw_val, nciw_info = metrics.compute_nciw(self.y_true, self.lower_narrow, self.upper_narrow, alpha=self.alpha_test, f=self.f_median)
        self.assertTrue(nciw_val >= 0)
        self.assertIn('c_test_cal', nciw_info)
        self.assertTrue(nciw_info['c_test_cal'] >= 0)

    def test_evaluate_intervals(self):
        eval_results = metrics.evaluate_intervals(self.y_true, self.lower_narrow, self.upper_narrow, alpha=self.alpha_test, f=self.f_median)
        self.assertIsInstance(eval_results, dict)
        expected_keys = ["PICP", "NIW", "MPIW", "QuantileLoss", "ExpectileLoss", "CRPS", "AUC", "NCIW", "IntervalScoreLoss", "c_test_cal"]
        for k in expected_keys:
            self.assertIn(k, eval_results)
            self.assertIsNotNone(eval_results[k]) # Should not be None, can be NaN for AUC/NCIW in edge cases
        
        # Test with mismatched lengths (should warn and truncate)
        with self.assertWarns(UserWarning):
             # This specific warning message is not standard, so we can't easily check it. 
             # We will check that UserWarning is raised.
             metrics.evaluate_intervals(self.y_true[:-1], self.lower_narrow, self.upper_narrow)

    def test_plot_auc_curve(self):
        # Test if it runs without error, actual plot verification is complex
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            metrics.plot_auc_curve(self.y_true, self.lower_narrow, self.upper_narrow, f=self.f_median, ax=ax)
            plt.close(fig)
        except ImportError:
            self.skipTest("Matplotlib not installed, skipping plot_auc_curve test")

if __name__ == '__main__':
    unittest.main() 