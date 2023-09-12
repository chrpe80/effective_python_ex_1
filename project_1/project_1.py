import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, f1_score
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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
        X = self.df.drop(self.target, axis=1)
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
        svr_param_grid = {"model__C": [0.1, 0.5, 1, 5, 10], "model__epsilon": [0.1, 0.5, 1, 2]}
        # Creating the cv models
        linear_reg = GridSearchCV(estimator=lr_pipe, param_grid=lr_param_grid, cv=10, scoring="r2")
        lasso = GridSearchCV(estimator=lasso_pipe, param_grid=lasso_param_grid, cv=10, scoring="r2")
        ridge = GridSearchCV(estimator=ridge_pipe, param_grid=ridge_param_grid, cv=10, scoring="r2")
        elastic_net = GridSearchCV(estimator=en_pipe, param_grid=en_param_grid, cv=10, scoring="r2")
        svr = GridSearchCV(estimator=svr_pipe, param_grid=svr_param_grid, cv=10, scoring="r2")
        # Fitting the models
        linear_reg.fit(X_train, y_train)
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        elastic_net.fit(X_train, y_train)
        svr.fit(X_train, y_train)
        # Make predictions
        lr_predictions = linear_reg.predict(X_test)
        lasso_predictions = lasso.predict(X_test)
        ridge_predictions = ridge.predict(X_test)
        en_predictions = elastic_net.predict(X_test)
        svr_predictions = svr.predict(X_test)
        # Evaluate lr
        lr_mae = mean_absolute_error(y_true=y_test, y_pred=lr_predictions)
        lr_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=lr_predictions))
        lr_r2 = linear_reg.score(X_test, y_test)
        # Evaluate lasso
        lasso_mae = mean_absolute_error(y_true=y_test, y_pred=lasso_predictions)
        lasso_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=lasso_predictions))
        lasso_r2 = lasso.score(X_test, y_test)
        # Evaluate ridge
        ridge_mae = mean_absolute_error(y_true=y_test, y_pred=ridge_predictions)
        ridge_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=ridge_predictions))
        ridge_r2 = ridge.score(X_test, y_test)
        # Evaluate elastic net
        en_mae = mean_absolute_error(y_true=y_test, y_pred=en_predictions)
        en_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=en_predictions))
        en_r2 = elastic_net.score(X_test, y_test)
        # Evaluate svr
        svr_mae = mean_absolute_error(y_true=y_test, y_pred=svr_predictions)
        svr_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=svr_predictions))
        svr_r2 = svr.score(X_test, y_test)
        # Create a dataframe with the model evaluations
        metrics_dict = {
            "Model": ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "SVR"],
            "MAE": [lr_mae, lasso_mae, ridge_mae, en_mae, svr_mae],
            "RMSE": [lr_rmse, lasso_rmse, ridge_rmse, en_rmse, svr_rmse],
            "R2 Score": [lr_r2, lasso_r2, ridge_r2, en_r2, svr_r2]
        }
        metrics_df = pd.DataFrame(metrics_dict)

        return linear_reg, lasso, ridge, elastic_net, svr, metrics_df

    def ann_regressor(self):
        X = self.df.drop(self.target, axis=1).values
        y = self.df[self.target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = Sequential()
        model.add(Dense(units=X.shape[1], activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(units=X.shape[1]*2, activation='relu'))
        model.add(Dense(units=X.shape[1]*2, activation='relu'))
        model.add(Dense(units=X.shape[1], activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        model.fit(x=X_train, y=y_train, epochs=500, callbacks=[early_stopping])
        predictions = model.predict(X_test)
        MAE = mean_absolute_error(y_test, predictions)
        RMSE = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        metrics_dict = {
            "Model": model,
            "MAE": MAE,
            "RMSE": RMSE,
            "R2 Score": r2
        }
        metrics_series = pd.Series(metrics_dict)

        return metrics_series, model

    def create_classification_models(self):
        # preparation
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Preprocessing
        scaler = StandardScaler()
        # Operations
        log_reg_operations = [("scaler", scaler), ("model", LogisticRegression(solver='saga', max_iter=5000))]
        knn_operations = [("scaler", scaler), ("model", KNeighborsClassifier())]
        svc_operations = [("scaler", scaler), ("model", SVC())]
        # Param grids
        log_reg_param_grid = {"model__penalty": ["l1", "l2"], "model__C": [0.001, 0.01, 0.1, 1, 10]}
        knn_param_grid = {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}
        svc_param_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": [0.1, 1, 10, 100]}
        # Pipes
        log_reg_pipe = Pipeline(log_reg_operations)
        knn_pipe = Pipeline(knn_operations)
        svc_pipe = Pipeline(svc_operations)
        # CV models
        log_reg = GridSearchCV(estimator=log_reg_pipe, param_grid=log_reg_param_grid, cv=10, scoring="f1_micro")
        knn = GridSearchCV(estimator=knn_pipe, param_grid=knn_param_grid, cv=10, scoring="f1_micro")
        svc = GridSearchCV(estimator=svc_pipe, param_grid=svc_param_grid, cv=10, scoring="f1_micro")
        # Model fitting
        log_reg.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        # Predictions
        log_reg_predictions = log_reg.predict(X_test)
        knn_predictions = knn.predict(X_test)
        svc_predictions = svc.predict(X_test)
        # f1 scores
        log_reg_score = f1_score(y_true=y_test, y_pred=log_reg_predictions, average="micro")
        knn_score = f1_score(y_true=y_test, y_pred=knn_predictions, average="micro")
        svc_score = f1_score(y_true=y_test, y_pred=svc_predictions, average="micro")
        scores_dict = {
            "log_reg": log_reg_score,
            "knn": knn_score,
            "svc": svc_score
        }
        scores_series = pd.Series(scores_dict)
        return scores_series, log_reg, knn, svc, log_reg_predictions, knn_predictions, svc_predictions

    def ann_classifier(self):
        nr_of_classes = self.df[self.target].nunique()
        if nr_of_classes < 3:
            loss = "binary_crossentropy"
            activation = "sigmoid"
        else:
            loss = "categorical_crossentropy"
            activation = "softmax"

        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def run(self):
        pass




























