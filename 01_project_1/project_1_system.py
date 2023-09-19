import pandas as pd
import unittest
from unittest.mock import patch
from project_1 import FindBestModel


class TestSystem(unittest.TestCase):
    def test_run_1(self):
        """Test the run method for regression models"""
        instance = FindBestModel()
        instance.model_type = "r"
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]

    def test_run_2(self):
        """Test the run method for binary classification models"""
        instance = FindBestModel()
        instance.model_type = "c"
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]

    def test_run_3(self):
        """	Test the run method for multi-class classification model"""
        instance = FindBestModel()
        instance.model_type = "c"
        instance.df = pd.read_csv("iris.csv")
        instance.target = "species"
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = ["1"]


if __name__ == "__main__":
    unittest.main()
