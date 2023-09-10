import pandas as pd
import os


class FindBestModel:
    def __init__(self):
        self.model_type = str()
        self.df = pd.DataFrame()
        self.target = str()

    def type_of_model(self):
        model_type = input("Enter type of model: ")
        if model_type in ["R".lower(), "C".lower()]:
            self.model_type = model_type
        else:
            raise ValueError("Wrong input")

    def read_in_csv(self):
        csv_file_path = input("Enter path to csv file: ")
        if os.path.isfile(csv_file_path):
            self.df = pd.read_csv(csv_file_path)
        else:
            raise ValueError("Wrong input")

    def print_out_column_names(self):
        for col in self.df.columns:
            print(col)

    def enter_target_column(self):
        target = input("Enter target: ")
        if target in self.df.columns:
            self.target = target
        else:
            raise ValueError(f"{target} doesn't seem to be among the columns")

    def check_if_continuous_or_categorical(self):
        nr_unique = self.df[self.target].nunique()
        threshold = 10
        if self.df[self.target].dtype == "object" or nr_unique <= threshold:
            return "Categorical"
        else:
            return "Continuous"

    def check_if_only_numerical_columns(self):
        numerical_columns = self.df.select_dtypes(exclude="object")
        if len(numerical_columns.columns) == len(self.df.columns):
            return True
        else:
            return False

    def check_if_missing_values(self):
        if self.df.isnull().sum().sum() == 0:
            return True
        else:
            return False

    def check_if_ready_for_ml(self):
        result_check_if_only_num = self.check_if_only_numerical_columns()
        result_check_if_missing_values = self.check_if_missing_values()
        report = str()
        if result_check_if_only_num and result_check_if_missing_values:
            return True
        else:
            if result_check_if_only_num and not result_check_if_missing_values:
                report = "This software only accepts numerical data"
            elif result_check_if_missing_values and not result_check_if_only_num:
                report = "There are missing values in your csv file"
            elif not result_check_if_only_num and not result_check_if_missing_values:
                report = "There are missing values and all columns are not numerical"

            return report

