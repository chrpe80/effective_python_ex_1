# Projekt 1

Create a python desktop app that makes it as easy as possible to find the best **regressor** or **classifier** for the data we get from our customers.

## Important

This project will be:

* In a new virtual environment
* In a git repo
* Pushed to a GitHub private repo with requirements.txt and .gitignore 
* Using OOP only
* Tested

## The project consists of these parts:

* User input:  enter the type of ml model

* Read in the csv file

* Print out all the column names 

* User input: enter the name of the target column

* Check if the target is continuous or categorical

* Check if the data is ready for the machine learning process or not

  * In case it is **not**, print a report that tells the user exactly what the problem was, and exit

  * In case it **is** and it's a **regressor** :

    * LiR
    * Lasso
    * Ridge
    * Elastic Net
    * SVR
    * ANN Deep learning model
    * A report for each regressor with: 
      * Best parameters (**Optional FOR ANN MODEL**)
      * MAE
      * RMSE
      * R2 score
      * A motivation as to why this is the best model for the data

  * In case it **is** and it's a classifier:

    * LoR
    * KNN
    * SVC
    * ANN
    * A report for each classifier with:
      * Best parameters (**Optional FOR ANN MODEL**)
      * Plot the confusion matrix for each model
      * Print classification report for each model
      * A motivation as to why this is the best model for the data

- At the end we need to ask the user if they agree with the feedback about the best ML model
  - If yes, then dump the model