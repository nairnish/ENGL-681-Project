# ENGL-681-Project

# Distinguishing between similar languages #

This repository consists of several experimentation carried out on the DSLCC dataset v4.0 for Distinguishing between similar language variants.

**Project Environment**
1. Pycharm IDE
2. Python Interpreter - Python 3.8

**Libraries/Packages used**
1. import nltk as nltk
2. import numpy as np
3. import pandas as pd
4. from sklearn.compose import ColumnTransformer
5. from gensim.parsing.preprocessing import STOPWORDS
6. from sklearn.feature_extraction.text import TfidfVectorizer
7. from sklearn.pipeline import Pipeline
8. nltk.download('punkt')
10. nltk.download('wordnet')
11. nltk.download('stopwords')
12. from nltk.corpus import stopwords
13. import time
14. import sys
15. from sklearn.metrics import classification_report

**Models Used**

****Basic Classification Models****

1. from xgboost import XGBClassifier
2. from sklearn.svm import LinearSVC
3. from sklearn.neighbors import KNeighborsClassifier
4. from sklearn.linear_model import LogisticRegression
5. from sklearn.naive_bayes import MultinomialNB
6. from sklearn.linear_model import SGDClassifier

****Requirements for Shallow Neural Network****

1. import matplotlib.pyplot as plt
2. from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
3. from sklearn.model_selection import train_test_split
4. from tensorflow.keras.models import Sequential
5. from tensorflow.keras import layers

****Requirements for Performing EDA - generating WordClouds, histogram****
1. from wordcloud import WordCloud
2. import pandas as pd
3. from PIL import Image
4. import plotly.graph_objects as go
5. import numpy as np
6. import matplotlib.pyplot as plt
7. from os import path
8. External Dependency - "mask.png" image required to be used as the mask image for wordcloud. The image is present in the project files.

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

#### 3. Evaluation  ####
Contains the code my_evaluation.py for evaluating the resuls from model predictions using the confusion matrix - F1, accuracy, precision and recall


