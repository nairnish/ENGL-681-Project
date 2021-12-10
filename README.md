# ENGL-681-Project

# Distinguishing between similar languages #

This repository consists of several experimentation carried out on the DSLCC dataset v4.0 for Distinguishing between similar language variants.

Below are the list of items present in each folder:

### Project Experiments ###
###### 1. Codes - Contains the model classification source codes and associated testing codes ######
1. SVC - The file test_SVC.py executes the linearSVC model in project_SVC.py
2. SGDC - The file test_SGDC.py executes the Stochastic Gradient Descent model in project_SGDC.py
3. MultinomialNB - The file test_NB.py executes the Multinomial Naive Bayes model in project_NB.py
4. XGBoost - The file test_XGBoost.py executes the Xtreme Gradient Boosting model in project_XGBoost.py
5. KNN - The file test_KNN.py executes the K-nearest neighbours model in project_KNN.py
6. LogisticRegression - The file test_LR.py executes the Logistic Regression model in project_LR.py
7. Shallow NeuralNetwork - The file sample_shallow_NN.py is WIP and is an implementation of shallow neural network

###### 2. DSLCC4 datastes ######
Contains the required datasets for the model to train, validate and test

###### 3. Evaluation  ######
Contains the code my_evaluation.py for evaluating the resuls from model predictions using the confusion matrix - F1, accuracy, precision and recall


