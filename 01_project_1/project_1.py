import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, \
    accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import os
from joblib import dump


class FindBestModel:
    def __init__(self):
        self.model_type = str()
        self.path = str()
        self.df = pd.DataFrame()
        self.target = str()

    def type_of_model(self):
        """
        Takes user input and sets model_type and if invalid input, raises ValueError
        :rtype: None
        """
        alternatives = ["[R]Regression", "[C]Classifier"]
        prompt = "\n".join(alternatives)
        model_type = input(f"{prompt}\n: ")
        if model_type.lower() in ["r", "c"]:
            self.model_type = model_type.lower()
        else:
            raise ValueError("Invalid input")

    def read_in_df(self):
        """
        Reads in the dataset if ready for ml
        :rtype: None
        """
        self.enter_path()
        self.check_if_only_numerical_columns()
        self.check_if_missing_values()
        self.df = pd.read_csv(self.path)

    def enter_path(self):
        """
        Validation of the path, raises ValueError if not a file
        :rtype: None
        """
        path = input("Enter path: ")
        if os.path.isfile(path):
            self.path = path
        else:
            raise ValueError("Not a file")

    def check_if_only_numerical_columns(self):
        """
        Validates if numerical columns only, raises ValueError if there are object columns
        :rtype: None
        """
        df = pd.read_csv(self.path)
        numerical_columns = df.select_dtypes(exclude="object")
        if len(numerical_columns.columns) == len(df.columns):
            pass
        else:
            raise ValueError("Not all columns are numerical")

    def check_if_missing_values(self):
        """
        Checks for missing values, raises ValueError if it has missing values
        :rtype: None
        """
        df = pd.read_csv(self.path)
        if df.isnull().sum().sum() == 0:
            pass
        else:
            raise ValueError("Has missing values")

    def read_in_target(self):
        """
        Reads in target and validates it, Raises ValueError if the target is not found
        :rtype: None
        """
        target = input("Enter target: ")
        in_df_columns = self.check_if_in_df_columns(target)
        if in_df_columns:
            self.target = target
        else:
            raise ValueError("Not found among columns")

    def check_if_in_df_columns(self, target):
        """
        Checks if the given target variable exists in the dataset columns
        :rtype: bool
        """
        if target in self.df.columns:
            return True
        else:
            return False

    @staticmethod
    def pipe_preparation(for_type, basic_model):
        """
        Prepares a pipeline for the given type of model
        :rtype: object
        """
        if for_type in ["LIR", "LASSO", "RIDGE", "EN"]:
            poly = PolynomialFeatures()
            scaler = StandardScaler()
            operations = [("poly", poly), ("scaler", scaler), ("model", basic_model)]
            pipe = Pipeline(operations)
            return pipe
        elif for_type in ["SVR", "LOR", "KNN", "SVC"]:
            scaler = StandardScaler()
            operations = [("scaler", scaler), ("model", basic_model)]
            pipe = Pipeline(operations)
            return pipe

    @staticmethod
    def param_grid_preparation(for_type):
        """
        Prepares a parameter grid for GridSearchCV based on the given type of mode
        :rtype: dict
        """
        if for_type == "LIR":
            param_grid = {"poly__degree": [2, 3, 4]}
            return param_grid
        elif for_type == "LASSO":
            param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
            return param_grid
        elif for_type == "RIDGE":
            param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
            return param_grid
        elif for_type == "EN":
            param_grid = {"poly__degree": [1, 2, 3, 4], "model__alpha": [0.2, 0.5, 1]}
            return param_grid
        elif for_type == "SVR":
            param_grid = {"model__C": [0.1, 0.5, 1, 5, 10], "model__epsilon": [0.1, 0.5, 1, 2]}
            return param_grid
        elif for_type == "LOR":
            param_grid = {"model__penalty": ["l1", "l2"], "model__C": [0.001, 0.01, 0.1, 1, 10]}
            return param_grid
        elif for_type == "KNN":
            param_grid = {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}
            return param_grid
        elif for_type == "SVC":
            param_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": [0.1, 1, 10, 100]}
            return param_grid

    def data_preparation(self, for_type):
        """
        Prepares the dataset for training and testing based on the given type of model
        :rtype: tuple
        """
        if for_type in ["LIR", "LASSO", "RIDGE", "EN", "SVR", "LOR", "KNN", "SVC", "ANN_REG", "ANN_CLASS_BIN"]:
            X = self.df.drop(self.target, axis=1)
            y = self.df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        elif for_type == "ANN_CLASS_MULTI":
            X = pd.get_dummies(self.df.drop(self.target, axis=1))
            y = pd.get_dummies(self.df[self.target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            nr_of_classes = self.df[self.target].nunique()
            return X_train, X_test, y_train, y_test, nr_of_classes

    def create_ml_regression_model(self, for_type, basic_model):
        # Get datasets
        X_train, X_test, y_train, y_test = self.data_preparation(for_type)
        # Get pipe
        pipe = self.pipe_preparation(for_type, basic_model)
        # Get param grid
        param_grid = self.param_grid_preparation(for_type)
        # Create grid_model
        grid_model = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring="r2")
        # Model fitting
        grid_model.fit(X_train, y_train)
        # Making predictions
        predictions = grid_model.predict(X_test)
        # Evaluations
        r2 = grid_model.score(X_test, y_test)
        MAE = mean_absolute_error(y_test, predictions)
        RMSE = np.sqrt(mean_squared_error(y_test, predictions))

        return grid_model, r2, MAE, RMSE

    def create_ann_regression_model(self):
        # Get datasets
        X_train, X_test, y_train, y_test = self.data_preparation("ANN_REG")
        # Scaling
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Create a Sequential model
        model = Sequential()
        model.add(Dense(units=X_train.shape[1], activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=X_train.shape[1], activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # Early stopping
        early_stopping = EarlyStopping(patience=10, mode="min")
        # Model fitting
        model.fit(X_train, y_train, epochs=500, validation_data=[X_test, y_test], callbacks=[early_stopping])
        # Predictions
        predictions = model.predict(X_test)
        # Evaluation
        r2 = r2_score(y_test, predictions)
        MAE = mean_absolute_error(y_test, predictions)
        RMSE = np.sqrt(mean_squared_error(y_test, predictions))

        return model, r2, MAE, RMSE

    def create_ml_classification_model(self, for_type, basic_model):
        # Get Datasets
        X_train, X_test, y_train, y_test = self.data_preparation(for_type)
        # Get pipe
        pipe = self.pipe_preparation(for_type, basic_model)
        # Get param grid
        param_grid = self.param_grid_preparation(for_type)
        # Create grid_model
        grid_model = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring="accuracy")
        # Model fitting
        grid_model.fit(X_train, y_train)
        # Making predictions
        predictions = grid_model.predict(X_test)
        # Evaluation
        accuracy_scr = accuracy_score(y_test, predictions)
        # Classification report
        print("-" * 60)
        print(f"Classification report for {type(basic_model).__name__}")
        print("-" * 60)
        print()
        print(classification_report(y_test, predictions))
        print()
        # Confusion matrix
        print("-" * 60)
        print(f"Confusion matrix for {type(basic_model).__name__}")
        print("-" * 60)
        print()
        print(confusion_matrix(y_test, predictions))
        print()
        # Best parameters
        print("-" * 60)
        print(f"Best parameters for {type(basic_model).__name__}")
        print("-" * 60)
        print()
        print(grid_model.best_params_)
        print()

        return grid_model, accuracy_scr

    def create_binary_ann_classification_model(self):
        # Get datasets
        X_train, X_test, y_train, y_test = self.data_preparation("ANN_CLASS_BIN")
        # Scaling
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Creating a model
        model = Sequential()
        model.add(Dense(units=X_train.shape[1], activation='relu'))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=1, activation="sigmoid"))
        # Compile the model
        model.compile(loss="binary_crossentropy", optimizer="adam")
        # Create EarlyStopping
        early_stopping = EarlyStopping(mode="min", patience=3)
        # Model fitting
        model.fit(X_train, y_train, epochs=500, validation_data=[X_test, y_test], callbacks=early_stopping)
        # Make predictions
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        # Evaluation
        accuracy_scr = accuracy_score(y_test, predictions)
        # Classification report
        print("-" * 60)
        print(f"Classification report for ANN")
        print("-" * 60)
        print()
        print(classification_report(y_test, predictions))
        print()
        # Confusion matrix
        print("-" * 60)
        print(f"Confusion matrix for ANN")
        print("-" * 60)
        print()
        print(confusion_matrix(y_test, predictions))
        print()

        return model, accuracy_scr

    def create_multi_ann_classification_model(self):
        # Get datasets
        X_train, X_test, y_train, y_test, nr_of_classes = self.data_preparation("ANN_CLASS_MULTI")
        # Scaling
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Creating a model
        model = Sequential()
        model.add(Dense(units=X_train.shape[1], activation='relu'))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=X_train.shape[1] * 2, activation='relu'))
        model.add(Dense(units=nr_of_classes, activation="softmax"))
        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        # Create EarlyStopping
        early_stopping = EarlyStopping(mode="min", patience=3)
        # Model fitting
        model.fit(X_train, y_train, epochs=500, validation_data=[X_test, y_test], callbacks=early_stopping)
        # Make predictions
        predictions = pd.get_dummies(np.argmax(model.predict(X_test), axis=1))
        # Evaluation
        accuracy_scr = accuracy_score(y_test, predictions)
        # Classification report
        print("-" * 60)
        print(f"Classification report for ANN")
        print("-" * 60)
        print()
        print(classification_report(y_test, predictions))
        print()
        # Confusion matrix
        print("-" * 60)
        print(f"Confusion matrix for ANN")
        print("-" * 60)
        print()
        print(confusion_matrix(y_true=y_test.idxmax(axis=1), y_pred=predictions.idxmax(axis=1)))
        print()

        return model, accuracy_scr

    @staticmethod
    def save(model_names, models, model_name):
        """
        Dumps a model
        :rtype: None
        """
        models_dict = dict(zip(model_names, models))
        dump(models_dict[model_name], f'saved_models/{model_name}.joblib')

    def run(self):
        """
        Executes the entire process
        :rtype: None
        """
        # Read in data
        self.type_of_model()
        self.read_in_df()
        self.read_in_target()

        # If the model type is regressor
        if self.model_type == "r":
            input_dict = {
                "LIR": LinearRegression(),
                "LASSO": Lasso(),
                "RIDGE": Ridge(),
                "EN": ElasticNet(),
                "SVR": SVR()
            }

            # Regression models and metrics
            reg_ml_models = []
            reg_ml_model_metrics = []
            reg_ml_model_names = ["linear_regression", "lasso", "ridge", "elastic_net", "SVR" "ANN"]
            for k, v in input_dict.items():
                model_and_metrics = self.create_ml_regression_model(k, v)
                model, r2, mae, rmse = model_and_metrics
                reg_ml_models.append(model)
                reg_ml_model_metrics.append((r2, mae, rmse))

            # ANN regression model
            ann_reg_model, r2, mae, rmse = self.create_ann_regression_model()
            reg_ml_models.append(ann_reg_model)
            reg_ml_model_metrics.append((r2, mae, rmse))

            # Preparing and printing a dataframe with metrics from all the regressors
            dict_of_metrics = dict(zip(reg_ml_model_names, reg_ml_model_metrics))
            index = ["r2", "MAE", "RMSE"]
            metrics_df = pd.DataFrame(dict_of_metrics, index=index).transpose().sort_values(by="r2", ascending=False)
            print("-" * 80)
            print("Results sorted by r2")
            print("-" * 80)
            print(metrics_df)
            print("-" * 80)
            print("Best parameters per model")
            print("-" * 80)

            # Preparing and printing a dataframe with the best parameters for the ml models
            model_names_best_params = ["Linear Regression", "Lasso", "Ridge", "Elastic Net"]
            best_params = [item.best_params_ for item in reg_ml_models if not type(item).__name__ == "Sequential"]
            best_params_df = pd.DataFrame(dict(zip(model_names_best_params, best_params))).transpose()
            best_params_df.columns = ["degree", "alpha"]
            print(best_params_df)

            # Making a recommendation based on R2 score
            print("-" * 80)
            print("Best model based on r2 score")
            print("-" * 80)
            print(f'The model that i recommend is {metrics_df["r2"].index[0]}, because it has the highest r2 score')
            print()

            # Checking if the client is happy with the recommendation and if so, dump the model
            prompt = "[1]accept\n[2]decline\n: "
            accept = input(prompt)
            if accept == "1":
                self.save(reg_ml_model_names, reg_ml_models, metrics_df["r2"].index[0])

            elif accept == "2":
                print("I am sorry hear that!")

            else:
                raise ValueError("Invalid input")

        # If the model type is classifier
        if self.model_type == "c":
            input_dict = {
                "LOR": LogisticRegression(solver="saga", max_iter=5000),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC()
            }

            cls_models = []
            cls_models_accuracy_scrs = []
            cls_models_names = ["logistic_regression", "KNN", "SVC", "ANN"]

            for k, v in input_dict.items():
                model, accuracy_scr = self.create_ml_classification_model(k, v)
                cls_models.append(model)
                cls_models_accuracy_scrs.append(accuracy_scr)

            if self.df[self.target].nunique() == 2:
                cls_ann_binary_model, cls_ann_binary_model_score = self.create_binary_ann_classification_model()
                cls_models.append(cls_ann_binary_model)
                cls_models_accuracy_scrs.append(cls_ann_binary_model_score)
                accuracy_scores_dict = dict(zip(cls_models_names, cls_models_accuracy_scrs))
                accuracy_scores_series = pd.Series(accuracy_scores_dict).sort_values(ascending=False)

                # Report
                print("-" * 60)
                print("Accuracy scores")
                print("-" * 60)
                print()
                print(accuracy_scores_series)
                print()
                print("-" * 60)
                print("Best model based on accuracy score")
                print("-" * 60)
                print()
                print(f"The recommended model is {accuracy_scores_series.index[0]} since it scored highest on accuracy")
                print()

                # Acceptance
                prompt = "[1]accept\n[2]decline\n: "
                accept = input(prompt)
                if accept == "1":
                    self.save(cls_models_names, cls_models, accuracy_scores_series.index[0])

                elif accept == "2":
                    print("I am sorry hear that!")

                else:
                    raise ValueError("Invalid input")

            elif self.df[self.target].nunique() > 2:
                cls_ann_multi_model, cls_ann_multi_model_score = self.create_multi_ann_classification_model()
                cls_models.append(cls_ann_multi_model)
                cls_models_accuracy_scrs.append(cls_ann_multi_model_score)
                accuracy_scores_dict = dict(zip(cls_models_names, cls_models_accuracy_scrs))
                accuracy_scores_series = pd.Series(accuracy_scores_dict).sort_values(ascending=False)

                # Report
                print("-" * 60)
                print("Accuracy scores")
                print("-" * 60)
                print()
                print(accuracy_scores_series)
                print()
                print("-" * 60)
                print("Best model based on accuracy score")
                print("-" * 60)
                print()
                print(f"The recommended model is {accuracy_scores_series.index[0]} since it scored highest on accuracy")
                print()

                # Acceptance
                prompt = "[1]accept\n[2]decline\n: "
                accept = input(prompt)
                if accept == "1":
                    self.save(cls_models_names, cls_models, accuracy_scores_series.index[0])

                elif accept == "2":
                    print("I am sorry hear that!")

                else:
                    raise ValueError("Invalid input")

            else:
                pass

# instance = FindBestModel()
