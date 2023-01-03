
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda,Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute, Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import regularizers, losses
from tensorflow.keras import backend as K
from functools import partial
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.framework import ops
import isolearn.keras as iso
import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import pandas as pd
import os
import pickle
import scipy.sparse as sp
import scipy.io as spio
import matplotlib.pyplot as plt
import isolearn.io as isoio
import isolearn.keras as isol


def iso_normalizer(array: np.ndarray) -> float:
    """
    Normalizes the given array by dividing the sum of elements from index 80 to index 104
    by the sum of all elements in the array. 
    
    Parameters
    ----------
    array: np.ndarray
        The array to be normalized
    
    Returns
    -------
    float
        The normalized value
    """
    iso_value = 0.0
    if np.sum(array) > 0.0 :
        iso_value = np.sum(array[80: 80+25]) / np.sum(array)
    
    return iso_value

def cut_normalizer(array: np.ndarray) -> np.ndarray:
    """
    Normalizes the given array by dividing each element by the sum of all elements in the array.
    
    Parameters
    ----------
    array: np.ndarray
        The array to be normalized
    
    Returns
    -------
    np.ndarray
        The normalized array
    """
    all_cuts = np.concatenate([np.zeros(205), np.array([1.0])])
    if np.sum(array) > 0.0 :
        all_cuts = array / np.sum(array)
    
    return all_cuts


def one_hot_encode_along_channel_axis(sequence: str) -> np.ndarray:
    """
    Encodes a DNA sequence as a one-hot array along the channel axis. From Av Shrikumar's code.
    
    Parameters
    ----------
    sequence: str
        The DNA sequence to encode
    
    Returns
    -------
    np.ndarray
        The one-hot encoded array
    """
    to_return = np.zeros((len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array: np.ndarray, sequence: str, one_hot_axis: int) -> None:
    """
    Fills in the given array with the one-hot encoding of the given DNA sequence. From Av Shrikumar's code.
    
    Parameters
    ----------
    zeros_array: np.ndarray
        The array to be filled in with the one-hot encoding
    sequence: str
        The DNA sequence to encode
    one_hot_axis: int
        The axis along which to encode the DNA sequence
    
    Returns
    -------
    None
    """
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    for i, char in enumerate(sequence):
        if char in ['A', 'a']:
            char_idx = 0
        elif char in ['C', 'c']:
            char_idx = 1
        elif char in ['G', 'g']:
            char_idx = 2
        elif char in ['T', 't']:
            char_idx = 3
        elif char in ['N', 'n']:
            continue
        else:
            raise RuntimeError(f"Unsupported character: {char}")
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1
