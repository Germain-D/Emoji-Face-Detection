# download FER dataset from https://www.kaggle.com/datasets/msambare/fer2013
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
import collections
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten,Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_main():

    train_dir = os.path.join("../data/fer/train")
    test_dir = os.path.join("../data/fer/test")

    fer_ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(48, 48),    # Resize images
        color_mode="grayscale", # Convert to grayscale (or use "rgb" if needed)
        batch_size=32,          # Set batch size
        label_mode="categorical",       # Use "categorical" for one-hot encoding
        shuffle=True,            # Shuffle the dataset
        validation_split=0.2,  # 20% of training data for validation
        subset="training",  # This dataset will be used for training
        seed=123
    )

    fer_ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(48, 48),    # Resize images
        color_mode="grayscale", # Convert to grayscale (or use "rgb" if needed)
        batch_size=32,          # Set batch size
        label_mode="categorical",       # Use "categorical" for one-hot encoding
        shuffle=True,            # Shuffle the dataset
        validation_split=0.2,  # Use the same validation split
        subset="validation",  # This dataset will be used for validation
        seed=123
    )


    fer_ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(48, 48),
        color_mode="grayscale",
        batch_size=32,
        label_mode="categorical",
        shuffle=False
    )

    for images, labels in fer_ds_train.take(1):
        print("Label shape:", labels.shape)  # Should be (batch_size, num_classes)
        print("Example label:", labels.numpy()[0])  # Should look like [0, 0, 1, 0, 0]

    class_names = fer_ds_train.class_names

    print("Class names:", class_names) 
    image_batch, label_batch = next(iter(fer_ds_train))  


    # Convert tensors to NumPy arrays
    image_batch_np = image_batch.numpy()
    label_batch_np = label_batch.numpy()

    # Randomly select 9 images from the batch
    random_indices = rnd.sample(range(len(image_batch_np)), 9)  # Pick 9 random indices
    random_images = [image_batch_np[i] for i in random_indices]
    random_labels = [class_names[np.argmax(label_batch_np[i])] for i in random_indices]  

    # Plot the images
    plt.figure(figsize=(10, 10))

    for i in range(9):  # Show 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(random_images[i].squeeze(), cmap="gray")  # Display grayscale image
        plt.title(random_labels[i])  # Show class name
        plt.axis("off")  # Hide axis

    plt.show()
    AUTOTUNE = tf.data.AUTOTUNE

    fer_ds_train = fer_ds_train.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    fer_ds_val = fer_ds_val.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
    fer_ds_test = fer_ds_test.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(7, activation='softmax')  # 7 classes for FER dataset
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Print model summary
    model.summary()

    earlystop = EarlyStopping(patience=5) 

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]
    history = model.fit(fer_ds_train, validation_data=fer_ds_val, epochs=10, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(fer_ds_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(accuracy))

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(epochs, accuracy, 'bo', label="Train_Acc")
    plt.plot(epochs, val_accuracy, 'r', label="val_Acc")
    plt.legend(loc='best', shadow=True)

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'bo', label="Train_Loss")
    plt.plot(epochs, val_loss, 'r', label="val_Loss")
    plt.legend(loc='best', shadow=True)

    plt.show()
    predictions = model.predict(fer_ds_test)
    print(predictions)
    # Convert probabilities to class indices
    predicted_labels = np.argmax(predictions, axis=1)

    # Get true labels
    true_labels = np.concatenate([y.numpy() for _, y in fer_ds_test])  
    true_labels = np.argmax(true_labels, axis=1)

    print(classification_report(true_labels, predicted_labels, zero_division=1))

    # Print an example
    print(f"Predicted Emotion: {class_names[predicted_labels[0]]}")
    print(f"True Emotion: {class_names[true_labels[0]]}")

    model.save("./models/CNN.keras")
    model.save_weights("./models/CNN.weights.h5")

if __name__ == "__main__":
    train_main()