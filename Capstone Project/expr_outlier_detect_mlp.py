from models.clf_mlp import MLP
import numpy as np
import sklearn
#from sklearn.preprocessing import
from data.preprocess import get_outlier_detection_set
import itertools

nb_categories = 2

def onehot(y):
    return np.array([[1 if i == j else 0 for i in range(nb_categories)] for j in y], dtype=np.float32)

if False :
    X_train, Y_train, X_test, Y_test = get_outlier_detection_set()
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    num_train, num_test = len(X_train), len(Y_test)
    num_features = np.prod(X_train.shape[1:])
    X_train = np.reshape(X_train, newshape=(num_train, num_features))
    X_test = np.reshape(X_test, newshape=(num_test, num_features))
    Y_train_onehot = onehot(Y_train)
else :
    X, Y = get_outlier_detection_set(split=False)
    num_features = np.prod(X.shape[1:])
    num_examples = X.shape[0]
    X = np.reshape(X, newshape=(num_examples, num_features))

n_hidden_options = [150,250,300]
l1_norm_options = [0, 1e-8]
l2_norm_options = [0,1e-7]
n_deep_options = [1,2,3]
learning_rate_options = [0.005, 0.0005] # 0.005, 0.001,
drop_options = [0.0]
optimizer_options = ['RMSprop'] #'SGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax'
activation_options = ['relu','sigmoid']
MAX_EPOCH = 65
PATIENCE = 3


hash_template = "n_hidden:{n_hidden}, l1_norm:{l1_norm}, l2_norm:{l2_norm}, n_deep:{n_deep}, drop:{drop}, learning_rate:{lr}, optimizer:{opt}"
all_results = {}


best_accuracy_yet = float('-inf')
best_acc_model_yet = ""
best_fbeta_yet = float('-inf')
best_fbeta_model_yet = ""

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
    Y_train_onehot = onehot(Y_train)

    for n_hidden, l1_norm, l2_norm, n_deep, drop, lr, opt, act in itertools.product(n_hidden_options, l1_norm_options, l2_norm_options, n_deep_options, drop_options, learning_rate_options, optimizer_options, activation_options) :
        model_hash = hash_template.format(n_hidden=n_hidden, l1_norm=l1_norm, l2_norm=l2_norm, n_deep=n_deep, drop=drop, lr=lr, opt=opt)

        print("Building model...")
    #    clf = build_model(num_feat, out_dim=out_dim, n_hidden=n_hidden, l1_norm=l1_norm,
    #                      l2_norm=l2_norm, n_deep=n_deep, drop=drop,
    #                      learning_rate=lr, optimizer=opt,
    #                      activation=act)

        clf = MLP(n_hidden=n_hidden, l1_norm=l1_norm,
                  l2_norm=l2_norm, n_deep=n_deep, drop=drop,
                  learning_rate=lr, optimizer=opt,
                  early_stop=False, max_epoch=MAX_EPOCH,
                  patience=PATIENCE, activation=act,
                  verbose=1)

        print("Model Hash : " + model_hash)

        print("Training model...")
        clf.fit(X_train, Y_train_onehot)#,X_test, Y_test)

        print("Calculating error on training set...")
        y_pred = clf.predict(X_train)
        print(np.sum(y_pred == Y_train))
        report = sklearn.metrics.classification_report(Y_train, y_pred)
        print(report)

        print("Calculating error on testing set...")
        y_pred = clf.predict(X_test)
        print(np.sum(y_pred == Y_test))
        report = sklearn.metrics.classification_report(Y_test, y_pred)
        score1 = sklearn.metrics.accuracy_score(Y_test, y_pred)
        score2 = sklearn.metrics.f1_score(Y_test, y_pred, average="weighted")
        # score = sklearn.metrics.fbeta_score(Y_test, y_pred, beta=0.5)
        fold_results[model_hash] = [score1, score2]
        print(report)
        if score1 > best_accuracy_yet:
            best_accuracy_yet = score1
            best_acc_model_yet = model_hash
        print("best_model_yet", best_acc_model_yet)
        print("best_score_yet", best_accuracy_yet)
        if score2 > best_fbeta_yet:
            best_fbeta_yet = score2
            best_fbeta_model_yet = model_hash
        print("best_model_yet", best_fbeta_model_yet)
        print("best_score_yet", best_fbeta_yet)
    all_folds.append(fold_results)
#Messy post processing
average_score_by_model = {}
all_models_hash = all_folds[0].keys()
for model_hash in all_models_hash:
    print(model_hash)
    results = map(lambda x: x[model_hash], all_folds)
    print(results)
    avg = np.average(results,axis=0)
    print(avg)
    average_score_by_model[model_hash] = avg

for metric_index in range(len(average_score_by_model.values()[0])):
    print('metric index',metric_index)

    best_score_yet = float('-inf')
    best_model_yet = None

    for model_hash in all_models_hash :
        score = average_score_by_model[model_hash][metric_index]
        if score > best_score_yet:
            best_score_yet = score
            best_model_yet = model_hash

    print("best_model ", best_model_yet)
    print("best_score ", best_score_yet)
    print(average_score_by_model[best_model_yet])
