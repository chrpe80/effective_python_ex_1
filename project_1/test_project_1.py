import pandas as pd
import unittest
from unittest.mock import patch
from project_1 import FindBestModel

def get_iris_label():
    df = pd.read_csv("iris.csv")
    df["species"] = df["species"].replace(["setosa", "versicolor", "virginica"], [0, 1, 2])
    return df

def get_iris_dummies():
    df = pd.get_dummies(pd.read_csv("iris.csv"))
    return df


class TestFindBestModel(unittest.TestCase):
    def test_type_of_model_wrong_input(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "U"
            with self.assertRaises(ValueError):
                instance.type_of_model()

    def test_type_of_model_correct_input(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "R".lower()
            instance.type_of_model()

    def test_read_in_csv_wrong_input(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "wrong input"
            with self.assertRaises(ValueError):
                instance.read_in_csv()

    def test_read_in_csv_valid_input(self):
        instance = FindBestModel()
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "advertising.csv"
            instance.read_in_csv()

    def test_print_out_column_names(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        with patch("builtins.print") as mocked_print:
            instance.print_out_column_names()
            self.assertGreater(mocked_print.call_count, 1)

    def test_enter_target_column_wrong_input(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "wrong input"
            with self.assertRaises(ValueError):
                instance.enter_target_column_reg()

    def test_enter_target_column_valid_input(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "sales"
            instance.enter_target_column_reg()

    def test_check_if_continuous_or_categorical(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        expected = ["Continuous", "Categorical"]
        self.assertIn(instance.check_if_continuous_or_categorical(), expected)

    def test_check_if_ready_for_ml_valid_input(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.check_if_only_numerical_columns()
        instance.check_if_missing_values()
        expectation = True
        self.assertEqual(expectation, instance.check_if_ready_for_ml())

    def test_check_if_ready_for_ml_wrong_input(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("ames.csv")
        expectation = "There are missing values and all columns are not numerical"
        self.assertEqual(expectation, instance.check_if_ready_for_ml())

    def test_create_reg_models(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertIsInstance(instance.create_reg_models(), tuple)
        self.assertEqual(len(instance.create_reg_models()), 6)

    def test_ann_regressor(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("advertising.csv")
        instance.target = "sales"
        self.assertIsInstance(instance.ann_regressor(), tuple)
        self.assertEqual(len(instance.ann_regressor()), 2)

    def test_create_classification_models(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        self.assertIsInstance(instance.create_classification_models(), tuple)
        self.assertEqual(len(instance.create_classification_models()), 7)

    def test_ann_classifier_binary(self):
        instance = FindBestModel()
        instance.df = pd.read_csv("hearing_test.csv")
        instance.target = "test_result"
        instance.classification_target_type = "B"
        self.assertIsInstance(instance.ann_classifier(), tuple)
        self.assertEqual(len(instance.ann_classifier()), 3)

    def test_ann_classifier_label(self):
        instance = FindBestModel()
        instance.df = get_iris_label()
        instance.target = "species"
        instance.classification_target_type = "L"
        self.assertIsInstance(instance.ann_classifier(), tuple)
        self.assertEqual(len(instance.ann_classifier()), 3)

    def test_ann_classifier_dummies(self):
        instance = FindBestModel()
        instance.df = get_iris_dummies()
        instance.target = "species_setosa,species_versicolor,species_virginica"
        instance.classification_target_type = "D"
        instance.nr_of_classes_one_hot_encoded = 3
        self.assertIsInstance(instance.ann_classifier(), tuple)
        self.assertEqual(len(instance.ann_classifier()), 3)


if __name__ == "__main__":
    unittest.main()





