1. store_features_in_file.py computes the HOG and Daisy features and also flattens the image.
2. train_validation*.py and train_test*.py consists a set of six python files
which implements  Logistic Regression, SVM and feed-forward neural network.
3. CNN and RNN named python files implements various CNN and RNN models.
4. neuralnet.py is our implemented feedforward neural network. To run a neural net, use neuralnet(netarch, training_data, test_data, learning rate)
    -> netarch is an array specifying the number of layers (including input & output layer) and number of neurons. 
        i.e. [3600,6,2,19] -> 4 layers in total: 3600 features in 1st layer, 6 neurons in 2nd layer, 2 neurons in the 3rd layer, 19 possible outputs in 4th/last layer 
    -> script included to take train.bin and create training_data and test_data