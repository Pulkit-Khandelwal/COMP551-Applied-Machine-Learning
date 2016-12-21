from models.clf_mlp import MLP
import numpy as np
import sklearn
#from sklearn.preprocessing import
from data.preprocess import get_outlier_detection_set
import itertools
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score


if True:
    X,Y = get_outlier_detection_set(split=False)
    num_train = len(X)
    num_features = np.prod(X.shape[1:])
    X = np.reshape(X, newshape=(num_train, num_features))
else :
    X_train, Y_train, X_test, Y_test = get_outlier_detection_set()
    num_train, num_test = len(X_train), len(Y_test)
    num_features = np.prod(X_train.shape[1:])
    print(num_features)
    X_train = np.reshape(X_train, newshape=(num_train, num_features))
    X_test = np.reshape(X_test, newshape=(num_test, num_features))
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

nb_categories = 2

names = ["Dummy_clf1", "Dummy_clf2",
         "Nearest Neighbors 1", "Nearest Neighbors 3",
         "Nearest Neighbors 5", "Nearest Neighbors 10",
         "Linear SVM C=0.025", "Sigmoid SVM C=1", "RBF SVM C=1", "balanced RBF SVM C=1",
         "Linear SVM C=0.75", "Sigmoid SVM C=2", "RBF SVM C=2", "balanced RBF SVM C=2",
         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis"]
 #        "Quadratic Discriminant Analysis"]
classifiers = [
    DummyClassifier(),
    DummyClassifier("most_frequent"),
    KNeighborsClassifier(1),
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="sigmoid"),
    SVC(gamma=2, C=1),
    SVC(gamma=2, C=1, class_weight='balanced'),
    SVC(kernel="linear", C=0.75),
    SVC(kernel="sigmoid", C=2),
    SVC(gamma=2, C=2),
    SVC(gamma=2, C=2, class_weight='balanced'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    GaussianNB(),
    AdaBoostClassifier(),
    LinearDiscriminantAnalysis()]


best_accuracy_yet = float('-inf')
best_model_yet = ""

all_folds = []

n_fold = 10
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=n_fold, test_size=.20, random_state=0)
assert rs.get_n_splits(X,Y)==n_fold

for train, test in rs.split(X):
    fold_results = {}
    print(train)
    print(test)
    X_train, Y_train = X[train], Y[train]
    X_test, Y_test = X[test], Y[test]

    for clfname, clf in zip(names,classifiers):

        print("Building model...")

        print("Model name : " + clfname)

        print("Training model...")
        clf.fit(X_train, Y_train)

        print("Calculating error on training set...")
        y_pred = clf.predict(X_train)
        print(np.sum(y_pred == Y_train))
        report = sklearn.metrics.classification_report(Y_train, y_pred)
        print(report)

        train_score1 = sklearn.metrics.accuracy_score(Y_train, y_pred)
        train_score2 = sklearn.metrics.f1_score(Y_train, y_pred, average="weighted")

        print("Calculating error on testing set...")
        y_pred = clf.predict(X_test)
#        print("good:", np.sum(y_pred == Y_test))
        report = sklearn.metrics.classification_report(Y_test, y_pred)

        score1 = sklearn.metrics.accuracy_score(Y_test, y_pred)
        score2 = sklearn.metrics.f1_score(Y_test, y_pred,average="weighted")
        #score = sklearn.metrics.fbeta_score(Y_test, y_pred, beta=0.5)
        print(report)


        fold_results[clfname] = [train_score1, train_score2, score1,score2]
    all_folds.append(fold_results)


# Messy post-processing

average_score_by_model = {}
for clfname in names:
    print(clfname)
    results = map(lambda x: x[clfname], all_folds)
    print(results)
    avg = np.average(results,axis=0)
    print(avg)
    average_score_by_model[clfname] = avg

models_to_print = names[:2]
for metric_index in range(len(average_score_by_model.values()[0])):
    print('metric index',metric_index)

    best_accuracy_yet = float('-inf')
    best_model_yet = None

    for clfname in names :
        score = average_score_by_model[clfname][metric_index]
        if score > best_accuracy_yet:
            best_accuracy_yet = score
            best_model_yet = clfname

    print("best_model_yet", best_model_yet)
    print("best_score_yet", best_accuracy_yet)
    models_to_print.append(best_model_yet)

for clfname in models_to_print:
    print(clfname)
    print(average_score_by_model[clfname])

for model_name, results in average_score_by_model.items():
    print(model_name + " & " + " & ".join(str(round(r,3)) for r in results)  + "\\\\")


#         + str(round(results[0],3)) +" & "+ str(round(results[1],3)) +"\\\\")





