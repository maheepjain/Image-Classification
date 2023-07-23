from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf


def logistic_regression_predict(X_train, y_train, X_test, y_test, model):
    # training the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy: ", score[1])
    return history
    
    
def cnn_predict(X_train, y_train, X_test, y_test, model):
    # compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # training the model
    history = model.fit(X_train, y_train, epochs=24, validation_data=(X_test, y_test))
    # evaluate the model
    score = model.evaluate(X_train, y_train, verbose=4)
    print("Test accuracy: ", score[1])
    return history


def random_forest_predict(X_train, y_train, X_test, y_test, model):
    # training model
    model.fit(X_train, y_train)
    # evaluate the model
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print('Test accuracy:', accuracy)
    # classification report
    print("Classification Report")
    print(classification_report(y_test, pred))
    # confusion matrix
    cm = confusion_matrix(y_test, pred)
    y_prob = model.predict_proba(X_test)
    # data for ROC and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return cm, fpr, tpr, roc_auc
