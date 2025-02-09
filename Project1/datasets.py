'''datasets.py
Loads and preprocesses datasets for use in neural networks.
Saad Khan and Nithun Selva
CS444: Deep Learning
'''
import tensorflow as tf
import numpy as np


def load_dataset(name):
    '''Uses TensorFlow Keras to load and return  the dataset with string nickname `name`.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.

    Returns:
    --------
    x: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set (preliminary).
    y: tf.constant. tf.int32s.
        The training set int-coded labels (preliminary).
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.

    Summary of preprocessing steps:
    -------------------------------
    1. Uses tf.keras.datasets to load the specified dataset training set and test set.
    2. Loads the class names from the .txt file downloaded from the project website with the same name as the dataset
        (e.g. cifar10.txt).
    3. Features: Converted from UINT8 to tf.float32 and normalized so that a 255 pixel value gets mapped to 1.0 and a
        0 pixel value gets mapped to 0.0.
    4. Labels: Converted to tf.int32 and flattened into a tensor of shape (N,).

    Helpful links:
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    '''
    
    # Load the dataset
    if name == 'cifar10':
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif name == 'mnist':
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        raise ValueError(f"Dataset name {name} not recognized.")
    
    # Load the class names
    with open(f'data/{name}.txt', 'r') as f:
        classnames = f.read().splitlines()

    # Convert to tf.float32 and normalize
    x = tf.constant(x, dtype=tf.float32) / 255.0
    x_test = tf.constant(x_test, dtype=tf.float32) / 255.0

    # Add a trailing color channel dimension if it doesn't exist
    if len(x.shape) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)

    # Convert labels to tf.int32
    y = tf.constant(y.flatten(), dtype=tf.int32)
    y_test = tf.constant(y_test.flatten(), dtype=tf.int32)

    return x, y, x_test, y_test, classnames


def standardize(x_train, x_test, eps=1e-10):
    '''Standardizes the image features using the global RGB triplet method.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    x_test: tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Test set features.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Standardized training set features (preliminary).
    tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Standardized test set features (preliminary).
    '''
    
    # Compute the mean and std of the training set
    mean = tf.reduce_mean(x_train, axis=(0, 1, 2))
    std = tf.math.reduce_std(x_train, axis=(0, 1, 2)) + eps

    # Standardize the training set
    x_train = (x_train - mean) / std
    
    # Standardize the test set
    x_test = (x_test - mean) / std

    return x_train, x_test


def train_val_split(x_train, y_train, val_prop=0.1):
    '''Subdivides the preliminary training set into disjoint/non-overlapping training set and validation sets.
    The val set is taken from the end of the preliminary training set.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    y_train: tf.constant. tf.int32s. shape=(N_train,).
        Training set class labels (preliminary).
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features.
    tf.constant. tf.int32s. shape=(N_train,).
        Training set labels.
    tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    '''
    
    # Compute the number of validation samples
    N_train = x_train.shape[0]
    N_val = int(np.round(N_train * val_prop))

    # Split the training set
    x_val = x_train[-N_val:]
    y_val = y_train[-N_val:]

    x_train = x_train[:-N_val]
    y_train = y_train[:-N_val]

    return x_train, y_train, x_val, y_val


def get_dataset(name, standardize_ds=True, val_prop=0.1):
    '''Automates the process of loading the requested dataset `name`, standardizing it (optional), and create the val
    set.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.
    standardize_ds: bool.
        Should we standardize the dataset?
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set.
    y_train: tf.constant. tf.int32s.
        The training set int-coded labels.
    x_val: tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    y_val: tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.
    '''
    
    # Load the dataset
    x_train, y_train, x_test, y_test, classnames = load_dataset(name)

    # Standardize the dataset
    if standardize_ds:
        x_train, x_test = standardize(x_train, x_test)

    # Split the training set into training and validation sets
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_prop=val_prop)

    return x_train, y_train, x_val, y_val, x_test, y_test, classnames
