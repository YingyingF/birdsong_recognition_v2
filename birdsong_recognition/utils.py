from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import tensorflow as tf
import random


__all__ = ['load_mp3', 'get_sample_labels', 'preprocess_file', 'pad_by_zeros', 'split_file_by_window_size',
           'wrapper_split_file_by_window_size', 'create_dataset_fixed_size', 'get_spectrogram', 'add_channel_dim',
           'dotdict']

# Cell
def load_mp3(file_label):
    """
    Reads and decodes mp3 based on file name.

    Args:
    file_label (list): audio file path and audio file label

    Returns:
    A tuple of decoded audio and label.
      
    """
    sample = tf.io.read_file(file_label[0])
    sample_audio = tfio.audio.decode_mp3(sample)
    return sample_audio, file_label[1]

def get_sample_labels(files, ebirds):
    """
    Converts label from string to numerical category.

    Args:
        files (list): list of audio file paths
        ebirds (list): string name of the birds

    Returns:
        [type]: [description]
    """
    labels = [i.split('/')[-2] for i in files]
    labels_num = [str(ebirds.index(i)) for i in labels]
    return list(zip(files, labels_num))

# Cell
def preprocess_file(sample_audio, label):
    """
    Minmax scale the audio signal.

    Args:
    sample_audio: decoded audio
    label (int): label of the sample_audio

    Returns:
    A tuple of scaled audio and int label.
    """
    # Only look at the first channel
    sample_audio = sample_audio[:,0]
    sample_audio_scaled = (sample_audio - tf.math.reduce_min(sample_audio))/(tf.math.reduce_max(sample_audio) - tf.math.reduce_min(sample_audio))
    sample_audio_scaled = 2*(sample_audio_scaled - 0.5)
    label = tf.cast(int(label), tf.int32)
    return sample_audio_scaled, label

# Cell
def pad_by_zeros(sample, min_file_size, last_sample_size):
    padding_size = min_file_size - last_sample_size
    sample_padded = tf.pad(sample, paddings=[[tf.constant(0), padding_size]])
    return sample_padded

# Cell
def split_file_by_window_size(sample, label, min_file_size=132300):
    # number of subsamples given none overlapping window size.
    subsample_count = int(np.round(sample.shape[0]/min_file_size))
    # ignore extremely long files for now
    subsample_limit = 75
    if subsample_count <= subsample_limit:
        # if the last sample is at least half the window size, then pad it, if not, clip it.
        last_sample_size = sample.shape[0]%min_file_size
        if last_sample_size/min_file_size > 0.5:
            sample = pad_by_zeros(sample, min_file_size, last_sample_size)
        else:
            sample = sample[:subsample_count*min_file_size]
        sample = tf.reshape(sample, shape=[subsample_count, min_file_size])
        label = tf.pad(tf.expand_dims(label, axis=0), paddings=[[0, subsample_count-1]], constant_values=label.numpy())
    else:
        sample = tf.reshape(sample[:subsample_limit*min_file_size], shape=[subsample_limit, min_file_size])
        label = tf.pad(tf.expand_dims(label, axis=0), paddings=[[0, 74]], constant_values=label.numpy())
    return sample, label

# Cell
def wrapper_split_file_by_window_size(sample, label, min_file_size=132300):
    """
    Format audio into 3 second segments.

    Args:
        sample (float32): decoded audio
        label (int): label
        min_file_size (int, optional): Minimum segment assuming audio sample rate of 44.1kHz. Defaults to 132300.

    Returns:
        Tuple: sample, label
    """
    sample, label = tf.py_function(split_file_by_window_size, inp=(sample, label, min_file_size),
            Tout=(sample.dtype, label.dtype))
    return sample, label

# Cell
def create_dataset_fixed_size(ds):
    iterator = iter(ds)
    sample, label = iterator.next()
    samples_all = tf.unstack(sample)
    labels_all = tf.unstack(label)

    while True:
        try:
            sample, label = iterator.next()
            sample = tf.unstack(sample)
            label = tf.unstack(label)
            samples_all = tf.concat([samples_all, sample], axis=0)
            labels_all = tf.concat([labels_all, label], axis=0)
        except:
            break
    return samples_all, labels_all

# Cell
def get_spectrogram(sample, label):
    spectrogram = tfio.experimental.audio.spectrogram(sample, nfft=512, window=512, stride=256)
    return spectrogram, label

# Cell
def add_channel_dim(sample, label):
    sample = tf.expand_dims(sample, axis=-1)
    return sample, label

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__