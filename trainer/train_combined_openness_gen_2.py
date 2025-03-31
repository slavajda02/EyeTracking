#!/usr/bin/env python
"""
Retrain Combined Eye Openness Model using a pre-existing model for relabeling.

This script loads eye images from an SQLite database, preprocesses and augments them,
and then uses an existing model to relabel the data. A new TensorFlow model is built,
trained on the relabeled data, and saved to the specified output directory.
"""

import os
import re
import argparse
import sqlite3
import base64
from io import BytesIO
import tempfile
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

# Global configuration constants.
BATCH_SIZE = 64
IMAGE_SIZE = 128       # Size to resize each eye image (width and height).
MAX_OFFSET = 10        # Maximum pixel offset for on-the-fly augmentation.
DEFAULT_LIMIT = 25000  # Default limit on the number of rows to query from the database.


def data_url_to_image(data_url: str) -> Image.Image:
    """
    Convert a data URL to a PIL Image.

    Parameters:
        data_url (str): The data URL containing the image encoded in base64.

    Returns:
        Image.Image: The decoded PIL Image.
    """
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data))


def preprocess_eye(data_url: str, size=(IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """
    Decode a data URL into a normalized image array.

    The image is converted to RGB, resized to the given size, and normalized to [0, 1].

    Parameters:
        data_url (str): The data URL of the eye image.
        size (tuple): Desired size (width, height) for the output image.

    Returns:
        np.ndarray: The processed image as a NumPy array.
    """
    try:
        img = data_url_to_image(data_url).convert("RGB").resize(size)
        img.load()  # Force load to trigger potential errors
        return np.array(img) / 255.0
    except Exception as e:
        raise ValueError(f"Error in preprocess_eye: {e}")


def random_offset_image(image: np.ndarray, max_offset: int = MAX_OFFSET) -> np.ndarray:
    """
    Apply a random offset to the image for augmentation.

    The image is shifted randomly along the x and y axes and empty regions are filled with zeros.

    Parameters:
        image (np.ndarray): The input image array.
        max_offset (int): Maximum offset (in pixels) for both x and y directions.

    Returns:
        np.ndarray: The augmented image with random offsets applied.
    """
    h, w, c = image.shape
    offset_x = np.random.randint(-max_offset, max_offset + 1)
    offset_y = np.random.randint(-max_offset, max_offset + 1)

    shifted = np.zeros_like(image)
    # Compute x-axis source and destination indices.
    if offset_x >= 0:
        src_x_start, src_x_end = 0, w - offset_x
        dest_x_start, dest_x_end = offset_x, w
    else:
        src_x_start, src_x_end = -offset_x, w
        dest_x_start, dest_x_end = 0, w + offset_x

    # Compute y-axis source and destination indices.
    if offset_y >= 0:
        src_y_start, src_y_end = 0, h - offset_y
        dest_y_start, dest_y_end = offset_y, h
    else:
        src_y_start, src_y_end = -offset_y, h
        dest_y_start, dest_y_end = 0, h + offset_y

    shifted[dest_y_start:dest_y_end, dest_x_start:dest_x_end, :] = (
        image[src_y_start:src_y_end, src_x_start:src_x_end, :]
    )
    return shifted


def load_data_from_db(db_path: str) -> tuple:
    """
    Internal helper to load eye image data and labels from the SQLite database.

    This function queries the database for rows with non-empty left and right eye frames,
    applies preprocessing and augmentation, and concatenates the left and right images side-by-side.

    Parameters:
        db_path (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of combined images.
            - np.ndarray: Array of corresponding labels.
    """
    query = f"""
        SELECT leftEyeFrame, rightEyeFrame, openness
        FROM training_data
        WHERE leftEyeFrame != ''
        AND rightEyeFrame != ''
        AND type == 'openness'
        ORDER BY RANDOM()
        LIMIT {DEFAULT_LIMIT}
    """
    print(f'Loading data from: {db_path}')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    combined_images = []
    labels = []
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    for left_frame, right_frame, openness in tqdm(rows):
        try:
            left_img = preprocess_eye(left_frame, size=target_size)
            right_img = preprocess_eye(right_frame, size=target_size)
            # Apply random offset augmentation to both images.
            left_img = random_offset_image(left_img, max_offset=MAX_OFFSET)
            right_img = random_offset_image(right_img, max_offset=MAX_OFFSET)
        except Exception as e:
            print("Skipping image due to error.")
            continue
        # Concatenate images horizontally.
        combined_img = np.concatenate([left_img, right_img], axis=1)
        combined_images.append(combined_img)
        labels.append([openness])

    return np.array(combined_images), np.array(labels)

def create_relabeling_dataset(data_dir: str, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset without shuffling for relabeling using the preprocessed images and labels.

    Parameters:
        data_dir (str): Path to the directory containing images and labels.
        batch_size (int): Batch size for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_path = os.path.join(data_dir, "labels.npy")

    # Load labels
    labels = np.load(labels_path)

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))]
    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    # Load and preprocess images
    def load_image(image_path):
        def load_and_cast(image_path):
            image = np.load(image_path.decode("utf-8"))  # Load the .npy file
            return image.astype(np.float32)  # Explicitly cast to float32

        image = tf.numpy_function(load_and_cast, [image_path], tf.float32)
        return image

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch the dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def relable_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """
    Relabel the dataset using the provided model.

    Parameters:
        model (tf.keras.Model): The model to use for relabeling.
        dataset (tf.data.Dataset): The dataset containing unshufled images without labels.

    Returns:
        np.ndarray: relabeled labels.
    """
    # Predict the labels using the model.
    predictions = model.predict(dataset, verbose=1)
    
    # Compute the 5th and 95th percentiles of the predictions.
    labels_flat = predictions.flatten()
    p5 = np.percentile(labels_flat, 5)
    p95 = np.percentile(labels_flat, 95)

    # Scale the predictions linearly so that the 5th percentile maps to 0 and the 95th to 0.75.
    new_training_labels_scaled = (predictions - p5) / (p95 - p5) * 0.75

    # Clip the scaled labels to be within [0, 0.75].
    new_training_labels_scaled = np.clip(new_training_labels_scaled, 0.0, 0.75)
    
    return new_training_labels_scaled

def preprocess_and_save_to_disk(model, db_path: str, output_dir: str) -> str:
    """
    Loads images from a database, applies offset augmentation relabels using an existing model
    and saves the images and labels as a .npy to a specified path.
    These files can then be used by the dataset loader.

    Parameters:
        model (tf.keras.Model): The existing model to use for relabeling.
        db_path (str): Path to the SQLite database file.
        output_dir (str): Directory to save the preprocessed images and labels.

    Returns:
        str: Path to the directory containing the saved images and labels.
    """
    images_path = os.path.join(output_dir, "images")
    labels_path = os.path.join(output_dir, "labels.npy")
    os.makedirs(images_path, exist_ok=True)
    
    # Load images and labels from the database
    combined_images, labels = load_data_from_db(db_path)
    
    # Save the images and  new labels to disk
    print("Saving preprocessed images and labels to disk...")
    for idx, combined_img in enumerate(tqdm(combined_images)):
        image_path = os.path.join(images_path, f"image_{idx}.npy")
        np.save(image_path, combined_img)
    np.save(labels_path, labels)
    
    # Use the model to predict the labels for the combined images.
    print("Relabeling training data using the loaded model...")
    relabeling_dataset = create_relabeling_dataset(output_dir)
    new_labels = relable_dataset(model, relabeling_dataset)
    print("Saving new labels to disk...")
    np.save(labels_path, new_labels)
    
    return output_dir

def create_tf_dataset(data_dir: str, batch_size: int = BATCH_SIZE, validation_split: float = 0.1) -> tuple:
    """
    Create TensorFlow datasets for training and validation from preprocessed images and labels.
    Used to reduce RAM usage.

    Parameters:
        data_dir (str): Path to the directory containing images and labels.
        batch_size (int): Batch size for the dataset.
        validation_split (float): Fraction of the dataset to use for validation.

    Returns:
        tuple: A tuple containing:
            - tf.data.Dataset: Training dataset.
            - tf.data.Dataset: Validation dataset.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_path = os.path.join(data_dir, "labels.npy")

    # Load labels
    labels = np.load(labels_path)

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))]
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))

    # Load and preprocess images
    def load_image(image_path, label):
        def load_and_cast(image_path):
            image = np.load(image_path.decode("utf-8"))  # Load the .npy file
            return image.astype(np.float32)  # Explicitly cast to float32

        image = tf.numpy_function(load_and_cast, [image_path], tf.float32)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and split the dataset
    dataset_size = len(image_files)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Batch and prefetch the datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

def build_model() -> tf.keras.Model:
    """
    Build and compile the Combined Eye Openness model.

    The model architecture consists of convolutional, pooling, and dense layers,
    and is compiled with the Adam optimizer and mean squared error loss.

    Returns:
        tf.keras.Model: The compiled TensorFlow Keras model.
    """
    model = Sequential([
        InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE * 2, 3)),
        
        Conv2D(32, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(64, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(128, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        
        Dense(64, activation='relu'),
        Dense(1, name='open-c')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Retrain Combined Eye Openness Model using a pre-existing model for relabeling"
    )
    parser.add_argument("--input_model_path", required=True, help="Path to an existing .h5 model file")
    parser.add_argument("--db_path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--output_dir", required=True, help="Directory where output files will be saved")
    args = parser.parse_args()

    input_model_path = args.input_model_path
    db_path = args.db_path
    output_dir = args.output_dir

    if not os.path.isfile(input_model_path):
        print("Model file not found. Exiting.")
        return
    if not os.path.isfile(db_path):
        print("Database file not found. Exiting.")
        return
    if not os.path.isdir(output_dir):
        print("Output directory not found. Exiting.")
        return
    
    # Load the existing model
    print("Loading existing model from:", input_model_path)
    loaded_model = tf.keras.models.load_model(input_model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    loaded_model.summary()
    
    # Get directory in the system temp folder to save data to
    temp_dir = tempfile.mkdtemp()
    print("Preprocessing and saving data to:", temp_dir)
    data_dir = preprocess_and_save_to_disk(loaded_model, db_path, temp_dir)
    print("Data loaded, relabeled and saved.")

    # Create TensorFlow datasets
    train_dataset, val_dataset = create_tf_dataset(data_dir)

    # Build and train the new model
    print("Building and training new model on relabeled data...")
    new_model = build_model()
    new_model.summary()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )

    new_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=250,
        callbacks=[lr_scheduler, early_stopping]
    )

    new_model_path = os.path.join(output_dir, "combined_openness_gen2.h5")
    new_model.save(new_model_path)
    print("New model saved to", new_model_path)
    
    #Remove the tmp directory
    print(f'Removing temp directory {temp_dir}')
    shutil.rmtree(temp_dir)
    print("Done!")
