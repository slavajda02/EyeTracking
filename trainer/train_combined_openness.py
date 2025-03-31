#!/usr/bin/env python
import os
# Set the variable before importing TensorFlow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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

MAX_OFFSET = 10              # Maximum pixel offset for on-the-fly augmentation.
BATCH_SIZE = 32             # Batch size for training.

def data_url_to_image(data_url: str):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data))

def preprocess_eye(data_url: str, size=(128, 128)):
    """Decode a data URL into an image array."""
    img = data_url_to_image(data_url).convert("RGB").resize(size)
    return np.array(img) / 255.0

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

def load_data_from_db(db_path: str):
    """
    Load combined eye images and openness labels from the SQLite database.
    
    Only rows where both leftEyeFrame and rightEyeFrame are non-empty are used.
    The combined image is created by concatenating the left and right images side-by-side.
    The label is a continuous value: openness.
    """
    print(f'Loading data from {db_path}')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT leftEyeFrame, rightEyeFrame, openness
        FROM training_data
        WHERE leftEyeFrame != '' AND rightEyeFrame != '' AND type == 'openness'
        ORDER BY RANDOM()
        LIMIT 25000
    """)
    rows = cursor.fetchall()
    conn.close()
    
    combined_images = []
    labels = []
    
    for left_frame, right_frame, openness in tqdm(rows):
        left_img = preprocess_eye(left_frame, size=(128, 128))
        right_img = preprocess_eye(right_frame, size=(128, 128))
        # Apply random offset augmentation to both images.
        left_img = random_offset_image(left_img, max_offset=MAX_OFFSET)
        right_img = random_offset_image(right_img, max_offset=MAX_OFFSET)
        # Concatenate images along width (axis=1)
        combined_img = np.concatenate([left_img, right_img], axis=1)
        combined_images.append(combined_img)
        labels.append([openness])
    
    return np.array(combined_images), np.array(labels)

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
            - tf.data.Dataset: Test dataset.
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
    test_size = int(val_size * 0.1)
    train_size = dataset_size - val_size
    

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    test_dataset = val_dataset.skip(test_size)

    # Batch and prefetch the datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def build_model():
    """
    Build and compile the Combined Eye Openness model.
    """
    model = Sequential([
        InputLayer(input_shape=(128, 256, 3)),
        
        Conv2D(32, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(64, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        
        Conv2D(128, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, name='open-c')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Train Combined Eye Openness Model from SQLite DB"
    )
    parser.add_argument("--db_path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--output_dir", required=True, help="Folder to save the trained model")
    args = parser.parse_args()
    
    # Get directory in the system temp folder to save data to
    temp_dir = tempfile.mkdtemp()
    print("Preprocessing and saving data to:", temp_dir)
    data_dir = preprocess_and_save_to_disk(args.db_path, temp_dir)
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset = create_tf_dataset(data_dir, validation_split=0.1)
    
    model = build_model()
    model.summary()
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=250,
        batch_size=32,
        callbacks=[lr_scheduler, early_stopping]
    )
    
    results = model.evaluate(test_dataset)
    print("Test loss and MAE:", results)
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "combined_openness.h5")
    model.save(model_save_path)
    print("Model saved to", model_save_path)

    #Remove the tmp directory
    print(f'Removing temp directory {temp_dir}')
    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == '__main__':
    main()