#!/usr/bin/env python
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
from tqdm import tqdm

# Global configuration constants.
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)      # Size to resize each eye image.
MAX_OFFSET = 10              # Maximum pixel offset for on-the-fly augmentation.
DEFAULT_LIMIT = 25000        # Default limit on the number of rows to query from the database.

def data_url_to_image(data_url: str):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data))

def preprocess_eye(data_url: str, size=IMAGE_SIZE):
    """Decode a data URL into an image array."""
    try:
        img = data_url_to_image(data_url).convert("RGB").resize(size)
        img.load()  # Force load to trigger potential errors
        return np.array(img) / 255.0
    except Exception as e:
        raise ValueError(f"Error in preprocess_eye: {e}")

def random_offset_image(image, max_offset=MAX_OFFSET):
    """
    Generate an augmented version of the image by randomly offsetting it in the x and y axes.
    Empty regions are filled with zeros.
    """
    h, w, c = image.shape
    offset_x = np.random.randint(-max_offset, max_offset + 1)
    offset_y = np.random.randint(-max_offset, max_offset + 1)
    
    shifted = np.zeros_like(image)
    # For x axis.
    if offset_x >= 0:
        src_x_start, src_x_end = 0, w - offset_x
        dest_x_start, dest_x_end = offset_x, w
    else:
        src_x_start, src_x_end = -offset_x, w
        dest_x_start, dest_x_end = 0, w + offset_x

    # For y axis.
    if offset_y >= 0:
        src_y_start, src_y_end = 0, h - offset_y
        dest_y_start, dest_y_end = offset_y, h
    else:
        src_y_start, src_y_end = -offset_y, h
        dest_y_start, dest_y_end = 0, h + offset_y
    
    shifted[dest_y_start:dest_y_end, dest_x_start:dest_x_end, :] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end, :]
    
    return shifted

def load_combined_data_from_db(db_path: str):
    """
    Load combined eye images and openness labels from the SQLite database.
    
    Only rows where both leftEyeFrame and rightEyeFrame are non-empty are used.
    The combined image is created by concatenating the left and right images side-by-side.
    The label is a continuous value: openness.
    Augmentation (random offset) is applied to each eye image.
    """
    print(f'Loading data from database: {db_path}')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT leftEyeFrame, rightEyeFrame, openness
        FROM training_data
        WHERE leftEyeFrame != ''
        AND rightEyeFrame != ''
        AND type == 'openness'
        ORDER BY RANDOM()
        LIMIT {DEFAULT_LIMIT}
    """)
    rows = cursor.fetchall()
    conn.close()
    
    combined_images = []
    labels = []
    left_images = []
    for left_frame, right_frame, openness in tqdm(rows):
        try:
            left_img = preprocess_eye(left_frame, size=IMAGE_SIZE)
            right_img = preprocess_eye(right_frame, size=IMAGE_SIZE)
            # Apply random offset augmentation to both images.
            left_img = random_offset_image(left_img, max_offset=MAX_OFFSET)
            right_img = random_offset_image(right_img, max_offset=MAX_OFFSET)
        except Exception as e:
            print("Skipping image due to error.")
            continue
        # Concatenate images along width (axis=1)
        combined_img = np.concatenate([left_img, right_img], axis=1)
        combined_images.append(combined_img)
        labels.append([openness])
        left_images.append(left_img)
    
    return np.array(combined_images), np.array(labels), np.array(left_images)

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
    
    # Compute the 5th and 95th percentiles of new_training_labels.
    # Since new_training_labels has shape (num_samples, 1), flatten it for percentile calculation.
    labels_flat = predictions.flatten()
    p5 = np.percentile(labels_flat, 5)
    p95 = np.percentile(labels_flat, 95)

    # Scale the labels linearly so that the 5th percentile maps to 0 and the 95th to 0.75.
    new_training_labels_scaled = (predictions - p5) / (p95 - p5) * 0.75

    # Clip the scaled labels so that values below 0 and above 0.75 are capped.
    new_training_labels_scaled = np.clip(new_training_labels_scaled, 0.0, 0.75)
    
    return new_training_labels_scaled

def preprocess_and_save_to_disk(db_path: str, output_dir: str) -> str:
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
    images_left_path = os.path.join(output_dir, "imagesleft")
    labels_path = os.path.join(output_dir, "labels.npy")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_left_path, exist_ok=True)
    
    # Load images and labels from the database
    combined_images, labels, right_images = load_combined_data_from_db(db_path)
    
    # Save the images and  new labels to disk
    print("Saving preprocessed images and labels to disk...")
    for idx, combined_img in enumerate(tqdm(combined_images)):
        image_path = os.path.join(images_path, f"image_{idx}.npy")
        np.save(image_path, combined_img)
    for idx, left_img in enumerate(tqdm(right_images)):
        left_image_path = os.path.join(images_left_path, f"imagesleft_{idx}.npy")
        np.save(left_image_path, left_img)
    np.save(labels_path, labels)
    
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
    images_dir = os.path.join(data_dir, "imagesleft")
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


def build_model():
    """
    Build and compile the Left Eye Openness model.
    This architecture matches train_left_openness.py.
    """
    model = Sequential([
        InputLayer(input_shape=(128, 128, 3)),
        
        Conv2D(16, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(32, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(64, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, name='open-c')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Train Left Eye Openness Model using combined eye model as a labeler"
    )
    parser.add_argument("--input_model_path", required=True, help="Path to an existing .h5 left openness model")
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
    
    # Determine the output path for the new model.
    new_model_path = os.path.join(output_dir, "left_openness_distilled.h5")
    
    # Load the existing model.
    print("Loading existing model from:", input_model_path)
    loaded_model = tf.keras.models.load_model(input_model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    loaded_model.summary()

    # Get directory in the system temp folder to save data to
    temp_dir = tempfile.mkdtemp()
    print("Preprocessing and saving data to:", temp_dir)
    data_dir = preprocess_and_save_to_disk(db_path, temp_dir)
    print("Data loaded and saved.")
    # Use the model to predict the labels for the combined images.
    print("Relabeling training data using the loaded model...")
    relabeling_dataset = create_relabeling_dataset(data_dir)
    new_labels = relable_dataset(loaded_model, relabeling_dataset)
    print("Saving new labels to disk...")
    labels_path = os.path.join(data_dir, "labels.npy")
    np.save(labels_path, new_labels)
    
    # Create TensorFlow datasets
    train_dataset, val_dataset = create_tf_dataset(data_dir)
    
    # Build a new model.
    print("Building and training new left eye model on relabeled data...")
    new_model = build_model()
    new_model.summary()
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    # Train the new model using training data sorted by predictions of the
    # combined eye model.
    new_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=250,
        callbacks=[lr_scheduler, early_stopping]
    )
    
    new_model.save(new_model_path)
    print("New left eye model saved to", new_model_path)
    
    #Remove the tmp directory
    print(f'Removing temp directory {temp_dir}')
    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == '__main__':
    main()
