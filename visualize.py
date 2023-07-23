import matplotlib.pyplot as plt
import seaborn as sns


def accuracy_cnn(history):
    # plotting entropy on a 3x1 grid, first subplot
    plt.subplot(3, 1, 1)
    # set title to entropy
    plt.title('CNN Entropy vs Epochs')
    # plotting entropy train vs test (train red test is blue)
    plt.plot(history.history['loss'], color='red', label='train')
    plt.plot(history.history['val_loss'], color='blue', label='test')
    # plot x and y labels on graph
    plt.xlabel('Epoch')
    plt.ylabel('Entropy loss')
    plt.legend(loc='best')
    # plotting accuracy on a 3x1 grid, second subplot
    plt.subplot(3, 1, 3)
    # set title to accuracy
    plt.title('CNN Accuracy vs Epochs')
    # plotting accuracy train vs test (train red test is blue)
    plt.plot(history.history['accuracy'], color='red', label='train')
    plt.plot(history.history['val_accuracy'], color='blue', label='test')
    # plot x and y labels on graph
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('accuracy_cnn.png')
    plt.close()
    return
    

def accuracy_logistic(history):
    # Plot accuracy vs epochs
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title('Logistic Regression Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_logistic.png')
    plt.close()
    return

def heatmap_confusion_matrix(cm):
    # plotting heatmap for confusion matrix
    ax = plt.axes()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax = ax)
    ax.set_title('Random Forest Heatmap')
    plt.savefig('heatmap.png')
    plt.close()
    return


def roc_curve_plot(fpr, tpr, roc_auc):
    # plot ROC curves for each class
    plt.figure()
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {} (AUC = {:.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    plt.close()
    return
