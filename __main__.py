from pre_processing import *
from build_features import *
from predict_model import *
from train_model import *
from visualize import *


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
