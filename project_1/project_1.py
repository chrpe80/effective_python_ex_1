import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

    def create_reg_models(self):
        # Preparation
        X = self.df.drop(self.target)
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Preprocessing
        poly = PolynomialFeatures()
        scaler = StandardScaler()
        # Operations variables
        lr_operations = [("poly", poly), ("scaler", scaler), ("model", LinearRegression())]
        lasso_operations = [("poly", poly), ("scaler", scaler), ("model", Lasso())]
        ridge_operations = [("poly", poly), ("scaler", scaler), ("model", Ridge())]
        en_operations = [("poly", poly), ("scaler", scaler), ("model", ElasticNet())]
        svr_operations = [("scaler", scaler), ("model", SVR())]
        # Pipes
        lr_pipe = Pipeline(lr_operations)
        lasso_pipe = Pipeline(lasso_operations)
        ridge_pipe = Pipeline(ridge_operations)
        en_pipe = Pipeline(en_operations)
        svr_pipe = Pipeline(svr_operations)
        # Param grids
        lr_param_grid = {"poly__degree": [1, 2, 3, 4]}
        lasso_param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
        ridge_param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
        en_param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
        svr_param_grid = {"C": [0.1, 0.5, 1, 5, 10], "epsilon": [0.1, 0.5, 1, 2]}
        # Creating the cv models
        linear_reg = GridSearchCV(estimator=lr_pipe, param_grid=lr_param_grid, cv=10, scoring="r2")
        lasso = GridSearchCV(estimator=lasso_pipe, param_grid=lasso_param_grid, cv=10, scoring="r2")
        ridge = GridSearchCV(estimator=ridge_pipe, param_grid=ridge_param_grid, cv=10, scoring="r2")
        elastic_net = GridSearchCV(estimator=en_pipe, param_grid=en_param_grid, cv=10, scoring="r2")
        svr = GridSearchCV(estimator=svr_pipe, param_grid=svr_param_grid, cv=10, scoring="r2")
        # Fitting the models
        linear_reg.fit(np.array(X_train), y_train)
        lasso.fit(np.array(X_train), y_train)
        ridge.fit(np.array(X_train), y_train)
        elastic_net.fit(np.array(X_train), y_train)
        svr.fit(np.array(X_train), y_train)
        # Make predictions
        lr_predictions = linear_reg.predict(np.array(X_test))
        lasso_predictions = lasso.predict(np.array(X_test))
        ridge_predictions = ridge.predict(np.array(X_test))
        en_predictions = elastic_net.predict(np.array(X_test))
        svr_predictions = svr.predict(np.array(X_test))
        # Evaluate lr
        lr_mae = mean_absolute_error(y_true=y_test, y_pred=lr_predictions)
        lr_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=lr_predictions))
        lr_r2 = linear_reg.score(np.array(X_test, y_test))
        # Evaluate lasso
        lasso_mae = mean_absolute_error(y_true=y_test, y_pred=lasso_predictions)
        lasso_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=lasso_predictions))
        lasso_r2 = lasso.score(np.array(X_test, y_test))
        # Evaluate ridge
        ridge_mae = mean_absolute_error(y_true=y_test, y_pred=ridge_predictions)
        ridge_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=ridge_predictions))
        ridge_r2 = ridge.score(np.array(X_test, y_test))
        # Evaluate elastic net
        en_mae = mean_absolute_error(y_true=y_test, y_pred=en_predictions)
        en_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=en_predictions))
        en_r2 = elastic_net.score(np.array(X_test, y_test))
        # Evaluate svr
        svr_mae = mean_absolute_error(y_true=y_test, y_pred=svr_predictions)
        svr_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=svr_predictions))
        svr_r2 = svr.score(np.array(X_test, y_test))
        # Create a dataframe with the model evaluations
        metrics_dict = {
            "Model": ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "SVR"],
            "MAE": [lr_mae, lasso_mae, ridge_mae, en_mae, svr_mae],
            "RMSE": [lr_rmse, lasso_rmse, ridge_rmse, en_rmse, svr_rmse],
            "R2 Score": [lr_r2, lasso_r2, ridge_r2, en_r2, svr_r2]
        }

        metrics_list = [metrics_dict]

        metrics_df = pd.DataFrame(metrics_list)



















