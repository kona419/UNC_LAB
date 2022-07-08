
# call packet for machine learning model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# read dataset 
data = pd.read_csv('IoT_data_set_v2_kny.csv')
print(data.shape)
data.head()
x_in = data.drop(columns=['ACTION']) # get all data in each column except the column with title "cap"
y_in = data['ACTION'] # get data in the column with title "cap"

x_train, x_test, y_train, y_test = train_test_split(x_in,y_in,
                                                    test_size = 0.15,
                                                 random_state = 0)


d=64

# DNN model for classification
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (3,)),
    keras.layers.Dense(d, activation='relu'),
    keras.layers.Dense(d, activation='relu'),
    keras.layers.Dense(d, activation='relu'),
 
    keras.layers.Dense(5, activation='softmax')
])


# the setting for training and compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)

#classifier = svm.SVC(kernel="linear", C=0.01).fit(x_train, y_train)

# how to define the output layer and data preparation?? 
est_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\n테스트 정확도:', test_acc)

#save the trained model
model.save("DNN_classification_UROP.h5")

#confusion matrix
y_true=[0,1,2,3,4]

y_pred = model.predict(x_test) #2500개
predicted = y_pred.argmax(axis=1)

print(confusion_matrix(y_test.values, predicted))


labels=['0', '1', '2', '3', '4'] 

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    #plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 예측값과 참값 
pred_labels = y_pred.argmax(axis=1)
true_labels = y_test.values

#메인 실행 
confusion_matrix = confusion_matrix(true_labels, pred_labels)
plot_confusion_matrix(confusion_matrix, labels=labels, normalize=True)
