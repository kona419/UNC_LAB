
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

# confusion matrix 그리는 함수 
def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=True):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    #plt.figsize=(100, 100)
    plt.title(title)
    plt.colorbar()    
    
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}'.format(labels[k])
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 1.5
    
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                #print('k:', con_mat[i, j])
                n = sum(con_mat[i])
                plt.text(j, i, '{:.2f}'.format(con_mat[i, j]/n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    
    
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
