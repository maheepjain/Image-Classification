from keras.utils import to_categorical
from data.pre_processing import load_cifar_testdata


def one_hot_encoding():
    # load the test data
    X_train, y_train, X_test, y_test = load_cifar_testdata()
    # convert labels to one-hot encoded vectors
    num_classes = 10
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return X_train, y_train, X_test, y_test


def flatten_dataset():
    # load the test data
    X_train, y_train, X_test, y_test = load_cifar_testdata()
    # flattening the images
    X_train = X_train.reshape(50000, 3 * 32 * 32)
    X_test = X_test.reshape(10000, 3 * 32 * 32)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, y_train, X_test, y_test
