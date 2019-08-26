import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, GRU
# from keras.optimizers import Adam
# from keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings("ignore") # shhhhhhhh


# Grid Search using Logistic Regression and CountVectorizer
def log_cvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('lr', LogisticRegression())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Grid search using Naive Bayes and CountVectorizer
def nae_vec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('nb', MultinomialNB())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score for the grid search was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Grid search through Logistic Regression after trasforming with TfidfVectorizer
def log_tfidf(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer()),
                     ('lr', LogisticRegression())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Grid search using Naive Bayes and CountVectorizer
def nae_tfidf(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer()),
                     ('nb', MultinomialNB())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score for the grid search was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Support Vector Machine. I had some major trouble with the runtime on this so ended up not using it.
def svm_cvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('svc', SVC(gamma='auto_deprecated'))])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Support Vector Machine. I had some major trouble with the runtime on this so ended up not using it.
def svm_tfidf(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer()),
                     ('svc', SVC(gamma='scale'))])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

def rand_for_cvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('rf', RandomForestClassifier())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

def rand_for_tfidf(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer()),
                     ('rf', RandomForestClassifier())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

# Extra Trees Classifier
def extra_tree_cvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer(stop_words = 'english')),
                 ('et', ExtraTreesClassifier())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

def extra_tree_tvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer(stop_words = 'english')),
                 ('et', ExtraTreesClassifier())])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))

def adaboost_cvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('cvec', CountVectorizer(stop_words = 'english')),
                     ('ada', AdaBoostClassifier(random_state=42))])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))


def adaboost_tvec(pipe_param, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('tvec', TfidfVectorizer(stop_words = 'english')),
                     ('ada', AdaBoostClassifier(random_state=42))])

    gs = GridSearchCV(pipe, param_grid=pipe_param, cv=5)
    gs.fit(X_train, y_train)
    train_predictions = gs.predict(X_train)
    test_predictions = gs.predict(X_test)
    print(f'The best score was: {gs.best_score_}')
    print(f'The accuracy score for your training data was: {accuracy_score(train_predictions, y_train)}')
    print(f'The accuracy score for your testing data was: {accuracy_score(test_predictions, y_test)}')
    print(f'The best parameters were: {gs.best_params_}')
    return(pd.DataFrame(confusion_matrix(y_test, test_predictions),
                                         index=['Actual Rational', 'Actual Irrational'],
                                         columns=['Predicted Rational', 'Predicted Irrational']))
# The name says it all
def zero_to_neg_one(value):
    if value == 0:
        return -1
    else:
        return value
