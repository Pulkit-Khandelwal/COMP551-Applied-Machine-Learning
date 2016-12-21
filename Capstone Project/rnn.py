import lasagne

def get_recurrent_net(input_var, modelconfig, exprconfig):
    """
    1-d recurrent network.

    Parameters
    ----------
    modelconfig - recurrence_type, {"lstm","rnn","gru"} : type of network.
                - grad_clip, float : max_value that can take the gradient.
                - output_size, int : the number of dimensions for the output.
                - num_hidden, int or list of int : the number of hidden nodes. if it is an iterable then layers will be stacked.
                - num_features : The size of the 1-d array in input at each timestep.
                - sequence_length, int : the number of timestep in an training example.

    exprconfig : batch_size

    Returns
    -------
    A recurrent network.
    """
    N_BATCH, SEQUENCE_LENGTH, NUM_FEATURES, N_HIDDEN, OUTPUT_SIZE, GRAD_CLIP = exprconfig["batch_size"], modelconfig[
        "sequence_length"], modelconfig["num_features"], modelconfig["num_hidden"], modelconfig["output_size"], \
                                                                          exprconfig["grad_clip"]
    if isinstance(N_HIDDEN, int):
        N_HIDDEN = [N_HIDDEN]

    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, SEQUENCE_LENGTH, NUM_FEATURES), input_var=input_var)
    l_forward = l_in

    if modelconfig["recurrence_type"] == "lstm" :
        for n in N_HIDDEN :
            l_forward = lasagne.layers.LSTMLayer(
                l_forward, n, grad_clipping=GRAD_CLIP)

    elif modelconfig["recurrence_type"] == "rnn":
        for n in N_HIDDEN:
            l_forward = lasagne.layers.RecurrentLayer(
                l_forward, n, grad_clipping=GRAD_CLIP,
                W_in_to_hid=lasagne.init.HeUniform(),
                W_hid_to_hid=lasagne.init.HeUniform(),
                nonlinearity=lasagne.nonlinearities.tanh)

    elif modelconfig["recurrence_type"] == "gru" :
        raise NotImplementedError()
    else :
        raise NotImplementedError()

    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1) #only the result is of interest
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=OUTPUT_SIZE, nonlinearity=lasagne.nonlinearities.tanh)
    return l_out