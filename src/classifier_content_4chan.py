import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

def vectorize(df):
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents='ascii')
    X = vectorizer.fit_transform(df['Reply'][:1000])
    y = np.append(
                  np.ones(len(df.sort_values('Compound Score', ascending=False)['Reply'].head(500))), 
                  np.zeros(len(df.sort_values('Compound Score', ascending=True)['Reply'].head(500)))
                 )
    return X, y

def get_dim(X, y):
    return X.shape, y.shape

def naive_bayes_classifier(X, y, test_size, rd_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rd_state)
    classifier = MultinomialNB()
    return X_train, X_test, y_train, y_test

def cross_validation_hyperparameters(X_train, y_train, parameter):
    param_grid = {
                    parameter : np.logspace(0,-9, num=100)
                }
    clf = GridSearchCV(MultinomialNB(), param_grid=param_grid, cv = 10, verbose=1, n_jobs=-1)
    best_classifier = clf.fit(X_train, y_train)
    return str(best_classifier.best_estimator_), best_classifier.score(X_train, y_train)

def cross_validation_predict(X_train, y_train, X_test, parameter):
    param_grid = {
                    parameter : np.logspace(0,-9, num=100)
                }
    clf = GridSearchCV(MultinomialNB(), param_grid=param_grid, cv = 10, verbose=1, n_jobs=-1)
    best_classifier = clf.fit(X_train, y_train)
    return best_classifier.predict(X_test)

def show_confusion_matrix(X_test, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_plot = ConfusionMatrixDisplay(cm, display_labels = ["Negative Polarity", "Positive Polarity"])
    cm_plot.plot()
    plt.show()

def performances_classifier(y_test, y_pred):
    return classification_report(y_test, y_pred), roc_auc_score(y_test, y_pred)