import tensorflow as tf
import cv2,os
import numpy as np
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(SCRIPT_DIR, "eval")
MASK_PATH = os.path.join(SCRIPT_DIR, "data", "with_mask")
NO_MASK_PATH = os.path.join(SCRIPT_DIR, "data", "without_mask")
IMG_SIZE = 50

def model_architecture():
    """Define the CNN architecture"""
    model = tf.keras.models.Sequential([
        # Convolutional layers with progression
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
        tf.keras.layers.BatchNormalization(),  # Stabilizes training
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # Extra layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        # Dense part
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Prevents overfitting
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def process_train_data(mask_path, no_mask_path):
    """Process images from both mask and no_mask directories"""
    images = []
    labels = []

    # Process images with mask (label = 1)
    for img in os.listdir(mask_path):
        img_array = cv2.imread(os.path.join(mask_path, img), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            continue
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_array = img_array / 255.0
        images.append(img_array)
        labels.append(1)

    # Process images without mask (label = 0)
    for img in os.listdir(no_mask_path):
        img_array = cv2.imread(os.path.join(no_mask_path, img), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            continue
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_array = img_array / 255.0
        images.append(img_array)
        labels.append(0)

    return np.array(images), np.array(labels)


def process_test_data(test_path):
    """Function to process test data"""
    images = []
    img_names = []
    for img in os.listdir(test_path):
        img_array = cv2.imread(os.path.join(test_path,img),cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        img_array = img_array/255.0
        images.append(img_array)
        img_names.append(img)
    return np.array(images), img_names


if __name__ == "__main__":
    model = model_architecture()
    images, labels = process_train_data(MASK_PATH, NO_MASK_PATH)
    history = model.fit(images.reshape(-1, IMG_SIZE, IMG_SIZE, 1), labels, epochs=5)
    test_images, img_names = process_test_data(TEST_PATH)
    predictions = model.predict(test_images.reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.show()