from sklearn.model_selection import GroupKFold
from data.preprocess import get_filtered_dataset
from models import reg_mlp
from lasagne.nonlinearities import sigmoid as sigm
import numpy as np


from theano import tensor as T
import theano
import lasagne

window_size = 2
#input_shape = (window_size, 79)
input_shape = (None,79*window_size)
nesterov_momentum = False
num_epochs = 3
depth = 2
output_radiant = True
num_folds = 5
batch_size = 32
if output_radiant :
    num_outputs = 2
else :
    num_outputs = 4

#input_var = T.tensor('inputs')
input_var = T.matrix('inputs')
target_var = T.matrix('targets')
#input_var = T.vector('inputs')
#target_var = T.vector('targets')

model = reg_mlp.build_model1(input_shape, num_output=num_outputs, num_units=200, input_var=input_var, depth=depth)#,nonlin=sigm)

print("Compiling functions")
output_var = lasagne.layers.get_output(model)
output_var = T.concatenate((output_var[0],np.pi*output_var[1]))
#test_output_var = lasagne.layers.get_output(model, deterministic=True) No dropout so isnt needed.


if False :
    loss = lasagne.objectives.squared_error(output_var, target_var)
    loss = loss.mean()
else :
    # assert_op = T.opt.Assert()
    # func = theano.function([output_var],assert_op(output_var,T.eq(output_var.size,2)))
    loss = (T.sin(output_var[1])*output_var[0] - T.sin(target_var[1])*target_var[0])**2 + (T.cos(output_var[1])*output_var[0] - T.cos(target_var[1])*target_var[0])**2
    loss = T.sqrt(loss).mean()

cost = theano.function([output_var, target_var], loss)

params = lasagne.layers.get_all_params(model, trainable=True)

if nesterov_momentum == True :
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)
else :
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.0005)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [loss])
predict_fn = theano.function([input_var],[output_var])


print("Retrieving dataset")
path_by_tag = get_filtered_dataset(window_size,output_radiant=output_radiant)
num_tags = len(path_by_tag.keys())
mapping = {i:j for i,j in enumerate(path_by_tag.keys())}

groups = []
g = 0
tmp_list = []
tmp_list_y = []
for tag, (X,Y) in path_by_tag.items():
    assert len(X) == len(Y)
    groups += [g for _ in range(len(X))]
    tmp_list.append(X)
    tmp_list_y.append(Y)
    g += 1

#X = np.array(tmp_list)
X = np.concatenate(tuple(tmp_list))
#X = np.reshape(X,newshape=(X.shape[0],X.shape[2]))
print(X.shape)
assert False
Y = np.concatenate(tmp_list_y)
print(Y.shape)

all_folds = []
group_kfold = GroupKFold(n_splits=num_folds)

model_hash = "hash"

for train_index, test_index in group_kfold.split(X,Y,groups):
    print(train_index,test_index)
    print(len(train_index))
    print(len(test_index))
    fold_result = {}
    X_train, Y_train = X[train_index], Y[train_index]
    X_test, Y_test = X[test_index], Y[test_index]

    train_loss = val_fn(X_train, Y_train)
    print("initial train loss : " + str(train_loss))
    validation_loss = val_fn(X_test, Y_test)
    print("initial validation loss : " + str(validation_loss))


    for epoch in range(num_epochs):
        print("epochs : "+str(epoch))
        ignored = 0
        train_loss = 0
        num_examples = len(X_train)
        for batch in range(0,num_examples,batch_size):
         #   print(batch)
            if batch*batch_size+batch_size >=num_examples:
#                raise Exception()
                ignored += (num_examples - batch*batch_size)
                continue
                train_loss += train_fn(X_train[batch*batch_size:num_examples], Y_train[batch*batch_size:num_examples])
            else :
                train_loss += train_fn(X_train[batch*batch_size:batch*batch_size+batch_size], Y_train[batch*batch_size:batch*batch_size+batch_size])

        train_loss *= train_loss/(num_examples-ignored)
   #     for x_train, y_train in zip(X_train,Y_train):
   #         print(y_train.shape)
   #         print(y_train)
   #         train_loss = train_fn(x_train, y_train)
#            train_loss = train_fn(x_train.flatten(), y_train.flatten())
        print("train loss : "+str(train_loss))
        validation_loss = val_fn(X_test, Y_test)
        print("validation loss : "+str(validation_loss))
        fold_result[model_hash] = (train_loss, validation_loss)
    all_folds.append(fold_result)

for metric_index in range(2):
    print(metric_index)
    print(np.average(map(lambda x:x[model_hash][metric_index],all_folds)))


