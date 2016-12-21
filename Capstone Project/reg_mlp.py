from lasagne import layers
from lasagne.nonlinearities import LeakyRectify as rect ,sigmoid as sigm


def build_model1(input_shape, num_output=2, input_var=None, depth=1, num_units=79, num_rbf=0, nonlin=rect):
    """ each layer has rbf and relu activation. all layers have the same distribution except the output.
    if dropout is True, it is set at 50% probability.
    """

    assert num_rbf <= num_units
    l = layers.InputLayer(input_shape,input_var=input_var)

    for d in range(depth):
        if num_rbf == 0:
            l = layers.DenseLayer(l,num_units=num_units, nonlinearity=nonlin())

        else :
            l1 = layers.DenseLayer(l,num_units=(num_units-num_rbf),nonlinearity=nonlin())
            l2 = layers.DenseLayer(l, num_units=num_rbf,nonlinearity=sigm)
            l = layers.ConcatLayer([l1,l2])
#        if dropout:
#            l = layers.DropoutLayer(l, p=0.5)

    if True :
        assert num_output== 2
        l2 = layers.DenseLayer(l, num_units=1, nonlinearity=nonlin())
        l1 = layers.DenseLayer(l, num_units=1, nonlinearity=sigm)
        l = layers.ConcatLayer([l1, l2])
    else :
        l = layers.DenseLayer(l,num_units=num_output)
    return l
