import numpy as np
import theano
import theano.tensor as T
import lasagne
import handlewav


def build_cnn(input_var=None):
    # input layer--takes a 7 x 252 matrix, 7 is the frame slices, 288 is the
    # possible frequency buckets (if we consider all 8 octaves)
    network = lasagne.layers.InputLayer(shape=(None, 1, 7, 288),
                                        input_var=input_var)

    # convolutional layer with 50 filters of size 5x25. stride=1, valid convolution
    # use tanh activation function (maybe try sigmoid or relu)
    # instantiate weights w/glorot process
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=50, filter_size=(5,25),
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

    # max-pool in the frequency only
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))

    # convolutional layer with 50 filters of size 3x5
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=50, filter_size=(3,5),
            nonlinearity=lasagne.nonlinearities.tanh)

    # max-pool in the frequency again
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))

    # Fully-connected layer with 1000 units, 50% dropout applied:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # Another Fully-connected layer with 200 units, 50% dropout again
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=200,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # Output layer of 88 outputs, transformed with sigmoid layer--not sure if
    # dropout should be implemented
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=88,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

# Taken from Lasagne's MNIST example
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
# Train the network, load data, etc. here
def main(num_epochs):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    # we want to minimize negative log-likelihood--i think we want to use cross entropy
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # update the parameters with SGD, try using a theano shared variable to decrease it linearly
    params = lasange.layers.get_all_params(network, trainable=True)
    learning_rates = np.linspace(0,0.01,1000)
    updates = lasange.updates.momentum(loss, params, learning_rate=learning_rates[len(learning_rates)-1], momentum=0.9)
    # updates = lasange.updates.momentum(loss, params, learning_rate=theano.shared(float32(0.01)), momentum=0.9)
    # alternatively maybe I could try something like:
    # alpha = theano.shared(float32(0.01))
    # updates = lasagne.updates.momentum(loss, params, learning_rate=alpha, momentum=0.9) #provide an update in the for loop

    # disable dropout for testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasange.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # compute performance on validation set
    acc = T.mean(T.eq(T.ceil(test_prediction - 0.5), target_var), dtype=theano.config.floatX)

    # Define training function. The way I currently update the learning
    # parameter means I may have to redefine this training function every time
    # I redefine the updates dictionary as well (unless it just grabs the one)
    # that I updated (maybe using a theano shared variable could fix that)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # train
    itr = 0
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # change the learning parameter
        # alpha.set_value(alpha - 0.01/1000.0)
        updates = lasange.updates.momentum(loss, params, learning_rate[len(learning_rate)-1-itr], momentum=0.9)
        # train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # call batch iterator: inputs should be a matrix of all the training inputs
        # targets should be an input matrix of all the targets
        # batchsize is 256
        # shuffle the dataset
        for batch in iterate_minibatches(train_inputs, train_targets, 256, shuffle=True):
            # set our input and target theano vars
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_inputs, val_targets, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        itr += 1

        print("Epoch {} of {} took {:.3f}s".format(epoch+1, num_epochs, time.time()-start_time))
        print("  Training loss:\t\t{:.6f}".format(train_err/train_batches))
        print("  Validation loss:\t\t{:.6f}".format(val_err/val_batches))
        print("  Validation acc:\t\t{:.2f}%".format(val_acc/val_batches * 100))

    # test after training
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_input, test_target, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err/test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc/test_batches * 100))

    # dump weights to file
    np.savez('weights.npz', *lasagne.layers.get_all_param_values(network))
