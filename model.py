import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scaling the data to improve accuracy
x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

print(model.evaluate(x_test, y_test))

y_predicted = model.predict(x_test)  # shape: (10000, 10)

# Convert softmax output to digit labels (0–9)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

print("Predicted labels:", y_predicted_labels[:10])
print("Actual labels:   ", y_test[:10])

cm = confusion_matrix(y_test, y_predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
