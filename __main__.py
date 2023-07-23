from data.pre_processing import load_cifar_testdata
from features.build_features import one_hot_encoding, flatten_dataset
from models.predict_model import cnn_predict, logistic_regression_predict, random_forest_predict
from models.train_model import cnn_model, logistic_regression_model, random_forest_model
from visualization.visualize import accuracy_cnn, accuracy_logistic, heatmap_confusion_matrix, roc_curve_plot


# Logistic Regression
print("#"*32)
print("Logistic Regression")
X_train, y_train, X_test, y_test = one_hot_encoding()
model = logistic_regression_model()
history = logistic_regression_predict(X_train, y_train, X_test, y_test, model)
accuracy_logistic(history)
print("#"*32)

# Random Forest
print("#"*32)
print("Random Forest")
X_train, y_train, X_test, y_test = flatten_dataset()
model = random_forest_model()
cm, fpr, tpr, roc_auc = random_forest_predict(X_train, y_train, X_test, y_test, model)
heatmap_confusion_matrix(cm)
roc_curve_plot(fpr, tpr, roc_auc)
print("#"*32)

# Convolutional Neural Network
print("#"*32)
print("Convolutional Neural Network")
X_train, y_train, X_test, y_test = load_cifar_testdata()
model = cnn_model()
history = cnn_predict(X_train, y_train, X_test, y_test, model)
accuracy_cnn(history)
print("#"*32)
