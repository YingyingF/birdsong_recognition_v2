from glob import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import tensorflow as tf
import random
from birdsong_recognition.utils import *

def preprocess(running_in_colab, ebirds):
    '''
    One time preprocessing.
    '''
    if running_in_colab:
        dataset_path = '../drive/MyDrive/dataset/'
    else:
        dataset_path = ''

    all_files = []

    for ebird in ebirds:
        files = glob(dataset_path + 'dataset/'+ebird+'/*')
        print('Number of files in {}: {}.'.format(ebird, len(files)))
        all_files.extend(files)

    random.shuffle(all_files)

    train_size = int(np.round(len(all_files) * 0.65))
    val_size = int(np.round(len(all_files) * 0.15))
    train_files = all_files[:train_size]
    val_files = all_files[train_size: train_size+val_size]
    test_files = all_files[train_size+val_size:]
    print('train_ds: {}. val_ds: {}. test_ds: {}'.format(len(train_files), len(val_files), len(test_files)))

    train_files = get_sample_labels(train_files, ebirds)
    val_files = get_sample_labels(val_files, ebirds)
    test_files = get_sample_labels(test_files, ebirds)

    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)

    train_ds = train_ds.map(load_mp3)
    val_ds = val_ds.map(load_mp3)
    test_ds = test_ds.map(load_mp3)

    train_ds = train_ds.map(preprocess_file)
    val_ds = val_ds.map(preprocess_file)
    test_ds = test_ds.map(preprocess_file)

    iterator = iter(train_ds)
    sample, label = iterator.next()
    assert sample.numpy().min() == -1
    assert sample.numpy().max() == 1

    shapes = []
    iterator = iter(train_ds)
    while True:
        try:
            sample,label = iterator.next()
            shapes.append(sample.shape[0])
        except:
            break

    min_file_size = np.array(shapes).min()
    max_file_size = np.array(shapes).max()
    print('min file size: {}, max file size: {}'.format(min_file_size, max_file_size))

    """### option 1: Using minimum sized file as the window size"""

    print('*'*30)
    print('Starting preprocessing with option 1')
    print('*'*30)

    train_win_ds = train_ds.map(wrapper_split_file_by_window_size)
    val_win_ds = val_ds.map(wrapper_split_file_by_window_size)
    test_win_ds = test_ds.map(wrapper_split_file_by_window_size)


    train_samples_all, train_labels_all = create_dataset_fixed_size(train_win_ds)
    val_samples_all, val_labels_all = create_dataset_fixed_size(val_win_ds)

    train_ds = tf.data.Dataset.from_tensor_slices((train_samples_all, train_labels_all))
    val_ds = tf.data.Dataset.from_tensor_slices((val_samples_all, val_labels_all))

    os.makedirs('preprocessed_dataset', exist_ok=True)
    tf.data.experimental.save(train_ds, 'preprocessed_dataset/train_ds', )
    tf.data.experimental.save(val_ds, 'preprocessed_dataset/val_ds', )


def create_ds(element_spec, return_audio_samples=False):
    '''
    Add spectrogram features
    '''
    train_ds = tf.data.experimental.load('preprocessed_dataset/train_ds', element_spec=element_spec)
    val_ds = tf.data.experimental.load('preprocessed_dataset/val_ds', element_spec=element_spec)

    if return_audio_samples:
        return train_ds, val_ds

    train_ds = train_ds.map(get_spectrogram)
    val_ds = val_ds.map(get_spectrogram)

    train_ds = train_ds.map(add_channel_dim)
    val_ds = val_ds.map(add_channel_dim)

    return train_ds, val_ds