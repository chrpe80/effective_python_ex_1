import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
import unittest
from unittest.mock import patch
from project_1 import FindBestModel



class TestFindBestModel(unittest.TestCase):
    def test_type_of_model_1(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "R"
            instance.type_of_model()
            self.assertEqual(instance.model_type, "r")

    def test_type_of_model_2(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "U"
            with self.assertRaises(ValueError):
                instance.type_of_model()

    def test_read_in_df_1(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "advertising"
            path = "advertising.csv"
            self.assertNotEqual(instance.path, path)
            with self.assertRaises(ValueError):
                instance.read_in_df()

    def test_read_in_df_2(self):
        instance = FindBestModel()
        expected_df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "advertising.csv"
            instance.read_in_df()
        self.assertTrue(instance.df.equals(expected_df))

    def test_enter_path(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "Invalid input"
            with self.assertRaises(ValueError):
                instance.enter_path()

    def test_check_if_only_numerical_columns(self):
        instance = FindBestModel()
        instance.path = "ames.csv"
        with self.assertRaises(ValueError):
            instance.check_if_only_numerical_columns()

    def test_check_if_missing_values(self):
        instance = FindBestModel()
        instance.path = "ames.csv"
        with self.assertRaises(ValueError):
            instance.check_if_missing_values()

    def test_read_in_target_1(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "invalid input"
            with self.assertRaises(ValueError):
                instance.read_in_target()

    def test_read_in_target_2(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "sales"
            instance.read_in_target()
            expectation = "sales"
            self.assertEqual(instance.target, expectation)

    def test_pipe_preparation(self):
        instance = FindBestModel()
        self.assertIsInstance(instance.pipe_preparation("LIR", LinearRegression()), Pipeline)

    def test_param_grid_preparation(self):
        instance = FindBestModel()
        self.assertEqual(instance.param_grid_preparation("LIR"), {"poly__degree": [2, 3, 4]})

    def test_data_preparation(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertEqual(len(instance.data_preparation("LIR")), 4)

    def test_create_ml_regression_model(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        basic_model = LinearRegression()
        self.assertEqual(len(instance.create_ml_regression_model("LIR", basic_model)), 4)

    def test_create_ann_regression_model(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertEqual(len(instance.create_ann_regression_model()), 4)

    def test_create_ml_classification_model(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        self.assertEqual(len(instance.create_ml_classification_model("LOR", LogisticRegression(solver="saga", max_iter=5000))), 2)

    def test_create_binary_ann_classification_model(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        self.assertEqual(len(instance.create_binary_ann_classification_model()), 2)

    def test_create_multi_ann_classification_model(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        self.assertEqual(len(instance.create_multi_ann_classification_model()), 2)

    def test_run_1(self):
        instance = FindBestModel()
        instance.model_type = "r"
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]

    def test_run_2(self):
        instance = FindBestModel()
        instance.model_type = "c"
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]

    def test_run_3(self):
        instance = FindBestModel()
        instance.model_type = "c"
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]


if __name__ == "__main__":
    unittest.main()













