# ENGL-681-Project

# Distinguishing between similar languages #

This repository consists of several experimentation carried out on the DSLCC dataset v4.0 for Distinguishing between similar language variants.

Below are the list of items present in each folder:

### Project Experiments ###
#### 1. Codes - Contains the model classification source codes and associated testing codes ####
1. SVC - The file test_SVC.py executes the linearSVC model in project_SVC.py
2. SGDC - The file test_SGDC.py executes the Stochastic Gradient Descent model in project_SGDC.py
3. MultinomialNB - The file test_NB.py executes the Multinomial Naive Bayes model in project_NB.py
4. XGBoost - The file test_XGBoost.py executes the Xtreme Gradient Boosting model in project_XGBoost.py
5. KNN - The file test_KNN.py executes the K-nearest neighbours model in project_KNN.py
6. LogisticRegression - The file test_LR.py executes the Logistic Regression model in project_LR.py
7. Shallow NeuralNetwork - The file shallow_NN.py is an implementation of shallow neural network

#### 2. DSLCC4 datastes ####

This is the training and test data for the Distinguishing between Similar Languages (DSL) task at VarDial 2017.

The package contains the following files:

1. DSL-TRAIN.txt - Training set for the DSL task
2. DSL-DEV.txt - Development set for the DSL task
3. DSL-TEST-UNLABELLED.txt - Unlabelled test set
4. DSL-TEST-GOLD.txt - Test set with gold labels

Each line in the .txt files are tab-delimited in the format:
sentence<tab>language-label

For more details (like data stats) you can refer to the VarDial 2017 task paper:

Marcos Zampieri, Shervin Malmasi, Nikola Ljubesic,
Preslav Nakov, Ahmed Ali, Jorg Tiedemann, Yves
Scherrer, and Noemi Aepli. 2017. "Findings of the
VarDial Evaluation Campaign 2017." In Proceedings
of the Fourth Workshop on NLP for Similar Languages,
Varieties and Dialects (VarDial), Valencia, Spain.


10/Feb/2017

#### 3. Evaluation  ####
Contains the code my_evaluation.py for evaluating the resuls from model predictions using the confusion matrix - F1, accuracy, precision and recall


