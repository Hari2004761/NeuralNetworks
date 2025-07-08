import keras
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train=x_train/255
x_test=x_test/255
# Plot the first 5 training images with labels
# for i in range(5,15):
#     plt.imshow(x_train[i])
#     plt.axis('off')
#     plt.show()

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

y_pred=model.predict(x_test)
y_predlabels=[int(np.argmax(i)) for i in y_pred]

print("Predicted labels: ", y_predlabels[:5])
print("Actual Labels: ",y_test[:5])

cm=confusion_matrix(y_test,y_predlabels)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=range(10))
disp.plot(cmap=plt.cm.Reds)
plt.title("Confusion Matrix of Fashion MNIST")
plt.show()
