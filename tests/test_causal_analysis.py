import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from script.causal_inference import perform_causal_analysis

class TestCausalAnalysis(unittest.TestCase):

    def setUp(self):
        # Sample data to simulate the merged data
        data = {
            'Trip End Time': pd.date_range(start='1/1/2022', periods=100, freq='H'),
            'datetime': pd.date_range(start='1/1/2022', periods=100, freq='H'),
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'is_weekend': np.random.randint(0, 2, 100),
            'driver_id': np.random.randint(1000, 1020, 100),
            'duration_hours': np.random.rand(100),
            'holiday': np.random.randint(0, 2, 100),
            'unfulfilled_requests': np.random.randint(0, 5, 100)
        }
        self.df = pd.DataFrame(data)

        self.train_df = self.df.sample(frac=0.8, random_state=42)
        self.holdout_df = self.df.drop(self.train_df.index)

    def test_data_loading(self):
        # Ensure data is loaded correctly
        self.assertFalse(self.df.empty)
        self.assertEqual(len(self.df), 100)

    def test_data_split(self):
        # Ensure data split correctly
        self.assertEqual(len(self.train_df), 80)
        self.assertEqual(len(self.holdout_df), 20)

    def test_causal_analysis(self):
        results = perform_causal_analysis(self.train_df)

        # Ensure results contain the expected interventions
        expected_interventions = ['driver_movement', 'order_location_accuracy', 'driver_operating_time', 'num_drivers']
        self.assertTrue(all(intervention in results for intervention in expected_interventions))

        # Ensure results are numeric
        self.assertTrue(all(isinstance(value, (int, float, np.number)) for value in results.values()))

    def test_results_with_mock_data(self):
        mock_results = {
            'driver_movement': 1.0,
            'order_location_accuracy': 0.5,
            'driver_operating_time': 2.0,
            'num_drivers': 1.5
        }
        with unittest.mock.patch('your_script.perform_causal_analysis', return_value=mock_results):
            results = perform_causal_analysis(self.train_df)
            self.assertEqual(results, mock_results)

if __name__ == '__main__':
    unittest.main()
