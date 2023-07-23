from tensorflow.keras import datasets


def load_cifar_testdata():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    # divide by 255 in order to make the data usable later
    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, y_train, X_test, y_test

