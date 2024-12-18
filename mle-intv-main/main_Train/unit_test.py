import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os

# Suponiendo que el código que has proporcionado está en un archivo llamado `model.py`
from train_model import load_data, preprocess_data, create_pipeline_data_processing, train_model, evaluate_model, save_model, load_model

class TestModelFunctions(unittest.TestCase):

    @patch('pandas.read_csv')  # Mocking pandas.read_csv
    def test_load_data(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'y': [1, 0, 1]
        })
        mock_read_csv.return_value = mock_df
        
        df = load_data("fake_path.csv")
        self.assertEqual(df.shape, (3, 3))  # Verifica que el dataframe tiene 3 filas y 3 columnas


if __name__ == '__main__':
    unittest.main()