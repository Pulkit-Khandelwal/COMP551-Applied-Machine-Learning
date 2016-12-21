from sklearn.model_selection import GroupKFold
from data.preprocess import get_filtered_dataset
from models import reg_mlp
import numpy as np


from theano import tensor as T
import theano
import lasagne

window_size = 1
#input_shape = (window_size, 79)
input_shape = (None,79*window_size)
output_radiant = True
if output_radiant :
    num_outputs = 2
else :
    num_outputs = 4


print("Retrieving dataset")
path_by_tag = get_filtered_dataset(window_size,output_radiant=output_radiant)
num_tags = len(path_by_tag.keys())

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

X = np.concatenate(tuple(tmp_list))
print(X.shape)
Y = np.concatenate(tmp_list_y)
print(Y.shape)

avg_loss = 0
for y in Y :
    print(y)
    l = np.sqrt((np.sin(y[1])*y[0])**2 + (np.cos(y[1])*y[0])**2)
    print(l)
    avg_loss += l

avg_loss *= 1.0/len(Y)
print(avg_loss)

