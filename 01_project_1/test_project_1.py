import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
import unittest
from unittest.mock import patch
from project_1 import FindBestModel


class TestFindBestModel(unittest.TestCase):
    def test_type_of_model_1(self):
        """Test if the model type is set to 'r' when user input is 'R'"""
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "R"
            instance.type_of_model()
            self.assertEqual(instance.model_type, "r")

    def test_type_of_model_2(self):
        """Test if a ValueError is raised when the user input is 'U'"""
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "U"
            with self.assertRaises(ValueError):
                instance.type_of_model()

    def test_read_in_df_1(self):
        """	Test if a ValueError is raised when the path is incorrect"""
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "advertising"
            path = "advertising.csv"
            self.assertNotEqual(instance.path, path)
            with self.assertRaises(ValueError):
                instance.read_in_df()

    def test_read_in_df_2(self):
        """Test if the DataFrame is read correctly from the given path"""
        instance = FindBestModel()
        expected_df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "advertising.csv"
            instance.read_in_df()
        self.assertTrue(instance.df.equals(expected_df))

    def test_enter_path(self):
        """	Test if a ValueError is raised when the user enters an invalid path"""
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "Invalid input"
            with self.assertRaises(ValueError):
                instance.enter_path()

    def test_check_if_only_numerical_columns(self):
        """Test if a ValueError is raised when the DataFrame contains non-numerical columns"""
        instance = FindBestModel()
        instance.path = "ames.csv"
        with self.assertRaises(ValueError):
            instance.check_if_only_numerical_columns()

    def test_check_if_missing_values(self):
        """	Test if a ValueError is raised when the DataFrame contains missing values"""
        instance = FindBestModel()
        instance.path = "ames.csv"
        with self.assertRaises(ValueError):
            instance.check_if_missing_values()

    def test_read_in_target_1(self):
        """	Test if a ValueError is raised when an invalid target column is provided"""
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "invalid input"
            with self.assertRaises(ValueError):
                instance.read_in_target()

    def test_read_in_target_2(self):
        """Test if the target column is correctly set when a valid column name is provided"""
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "sales"
            instance.read_in_target()
            expectation = "sales"
            self.assertEqual(instance.target, expectation)

    def test_pipe_preparation(self):
        """Test if a Pipeline object is returned for a given model type"""
        instance = FindBestModel()
        self.assertIsInstance(instance.pipe_preparation("LIR", LinearRegression()), Pipeline)

    def test_param_grid_preparation(self):
        """Test if the correct parameter grid is returned for a given model type"""
        instance = FindBestModel()
        self.assertEqual(instance.param_grid_preparation("LIR"), {"poly__degree": [2, 3, 4]})

    def test_data_preparation(self):
        """Test if the data is correctly prepared for a given model typ"""
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertEqual(len(instance.data_preparation("LIR")), 4)

    def test_create_ml_regression_model(self):
        """Test if the machine learning regression model is correctly created"""
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        basic_model = LinearRegression()
        self.assertEqual(len(instance.create_ml_regression_model("LIR", basic_model)), 4)

    def test_create_ann_regression_model(self):
        """Test if the artificial neural network regression model is correctly created"""
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertEqual(len(instance.create_ann_regression_model()), 4)

    def test_create_ml_classification_model(self):
        """Test if the 'machine learning classification model is correctly created"""
        instance = FindBestModel()
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        self.assertEqual(
            len(instance.create_ml_classification_model("LOR", LogisticRegression(solver="saga", max_iter=5000))), 2)

    def test_create_binary_ann_classification_model(self):
        """Test if the binary artificial neural network classification model is correctly created"""
        instance = FindBestModel()
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        self.assertEqual(len(instance.create_binary_ann_classification_model()), 2)

    def test_create_multi_ann_classification_model(self):
        """Test if the multi-class artificial neural network classification model is correctly created"""
        instance = FindBestModel()
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        self.assertEqual(len(instance.create_multi_ann_classification_model()), 2)


if __name__ == "__main__":
    unittest.main()
