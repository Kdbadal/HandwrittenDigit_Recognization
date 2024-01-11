import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path for saving and loading the model
model_path = 'handwritten_digits.model'

# Decide if to load an existing model or to train a new one
train_new_model = not os.path.exists(model_path) or os.path.getsize(model_path) == 0

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=10)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save(model_path)
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)

# Load custom images and predict them
image_number = 2
while os.path.isfile(f'/home/kush/handwrittenDigit_Recognization/images/digit{image_number}.png'):
    try:
        img = cv2.imread(f'/home/kush/handwrittenDigit_Recognization/images/digit{image_number}.png')[:, :, 0]

        # Resize the image to the expected input size (28x28)
        img = cv2.resize(img, (28, 28))

        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except Exception as e:
        print(f"Error reading image {image_number}: {e}")
        print("Proceeding with the next image...")
        image_number += 1
