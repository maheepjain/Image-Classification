from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers, models


def logistic_regression_model():
    # make a sequential model
    model = models.Sequential()
    # adding dense layer
    model.add(layers.Dense(units=10, activation='softmax', input_shape=(3072,)))
    # compiling the model 
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def cnn_model():
    # make a sequential model
    model = models.Sequential()
    # add the convolution layer, and then add the pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # dropout normalization rate is 20%
    model.add(layers.Dropout(0.2))
    # repeat adding convolution, pooling, and dropout 2 more times
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # flatten 
    model.add(layers.Flatten())
    # add 2 more dense layers on top
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    # print the summary of the model
    model.summary()
    return model


def random_forest_model():
    # building a random forest model
    rf = RandomForestClassifier(random_state=42)
    # providing tuning parameters
    param_grid = { 
        'n_estimators': [75, 100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 100, 150, None],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False]
        }

    # using grid search to find the best tuning paramets
    GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    max_depth_best_param_ = 10
    criterion_best_param_ = 'gini'
    min_samples_leaf_best_param_ = 10
    # training model with best parameters
    model = RandomForestClassifier(criterion=criterion_best_param_, max_depth=max_depth_best_param_, min_samples_leaf=min_samples_leaf_best_param_)
    return model

