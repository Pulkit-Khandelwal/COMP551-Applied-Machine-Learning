from models.deepbelief.code.deepbelief.rbm import RBM
from models.deepbelief.code.deepbelief.gaussianrbm import GaussianRBM
from models.deepbelief.code.deepbelief.dbn import DBN
from data import preprocess
from sklearn.model_selection import GroupKFold

def main():

    X, groups = preprocess.get_dataset(4)
    examples_indexes = [i for i in range(len(X))]
    group_kfold = GroupKFold(n_splits=3)
    # group_kfold.get_n_splits(X, y, groups)


    for train_indexes, test_indexes in group_kfold.split(examples_indexes, groups=groups):
        #print(train_indexes, test_indexes)
        X_train, X_test = X[train_indexes], X[test_indexes]
        rbm = train_RBM(X_train ,20,30)

        #    for x in X_test:
        metric = rbm.estimate_log_likelihood(X_train.T)
        print(metric)
        metric = rbm.estimate_log_likelihood(X_test.T)
        print(metric)

from math import ceil

def train_RBM(X, num_hiddens, num_epochs, batch_size=16,type="gaussian"):
    assert len(X.shape) == 2
    num_visibles = X.shape[1]
  #  print(num_visibles,num_hiddens)
    if type == "gaussian":
        rbm = GaussianRBM(num_visibles,num_hiddens)
    else :
        rbm = RBM(num_visibles,num_hiddens)
    rbm.learning_rate = 1E-3
    rbm.momentum = 0.8
    rbm.weight_decay = 1E-2

    num_examples = len(X)
    for epoch in range(num_epochs):
        if True : #Batch
            for batch in range(int(ceil(num_examples/float(batch_size)))):
    #            print("batch", batch)
                ind = batch*batch_size
                if ind+batch_size >= num_examples:
                    X_batch = X[ind:]
                else :
                    X_batch = X[ind:ind+batch_size]
            rbm.train(X_batch.T)
        else :
            for x in X :
                rbm.train(x)
    return rbm


def main2():
    X, groups = preprocess.get_dataset(4)
    examples_indexes = [i for i in range(len(X))]
    group_kfold = GroupKFold(n_splits=2)
    # group_kfold.get_n_splits(X, y, groups)


    for train_indexes, test_indexes in group_kfold.split(examples_indexes, groups=groups):
        print(train_indexes, test_indexes)
        X_train, X_test = X[train_indexes], X[test_indexes]
        dbn = train_dbn(X_train.T)
        samples = dbn.sample(num_samples=2, burn_in_length=100, sample_spacing=20, num_parallel_chains=1)

        print(samples)
        print(len(samples))
        print(samples.shape)
        metric = dbn.estimate_log_likelihood(X_train)

        print(metric)
        metric = dbn.estimate_log_likelihood(X_test)
        print(metric)

        samples = dbn.sample(num_samples=1, burn_in_length=100, sample_spacing=20, num_parallel_chains=1)
        print(samples)

def train_dbn(X, batch_size=32,num_epochs=50):
    #num_visibles = X.shape[0]
    num_hiddens = [500,500]
    # train 1st layer
    rbm = train_RBM(X, num_hiddens[0], num_epochs, batch_size, type="agaussian")
    dbn = DBN(rbm)
    #dbn = DBN(GaussianRBM(num_visibles, num_hiddens[0]))
    dbn[0].learning_rate = 1E-3
    dbn[0].momentum = 0.8
    dbn[0].weight_decay = 1E-2
#    dbn.train(X.T, num_epochs, batch_size)
    return dbn

if __name__=="__main__":
    main2()