import tensorflow as tf
import random
import argparse
import tensorflow.keras as keras
from tensorflow.keras import layers, losses, optimizers
import wandb as wb
from wandb.keras import WandbCallback

from birdsong_recognition.preprocess import *
from birdsong_recognition.utils import *


def main(config):
    seed = 42
    random.seed(seed)

    run_test = config.run_test

    element_spec = (tf.TensorSpec(shape=(132300,), dtype=tf.float32, name='input'),
                    tf.TensorSpec(shape=(), dtype=tf.int32, name='label'))
    train_ds, val_ds = create_ds(element_spec)
    
    """### simple model"""
    keras.backend.clear_session()

    model = tf.keras.Sequential([
                layers.Conv2D(filters=32, kernel_size=(4,4), strides=1, activation='relu', input_shape=(517, 257, 1)),
                layers.MaxPool2D(pool_size=(4,4)),
                layers.Conv2D(filters=64, kernel_size=(4,4), strides=1, activation='relu'),
                layers.MaxPool2D(pool_size=(4,4)),
                layers.Flatten(),
                layers.Dense(config.hidden_layer_size, activation='relu'),
                layers.Dropout(config.dropout),
                layers.Dense(3)
    ])

    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr,), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_ = train_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)
    val_ds_ = val_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)

    print('*'*30)
    print('Starting training')
    print('*'*30)

    if run_test:
        model.fit(train_ds_, epochs=2, validation_data=val_ds_, steps_per_epoch=2)
    else:
        model.fit(train_ds_, epochs=config.epochs, validation_data=val_ds_, callbacks=[WandbCallback()])       

    model.save('model.h5')
    wb.finish()

if __name__== '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_test', type=bool, default=False)
    arg_parser.add_argument('--lr', default=1e-3)
    arg_parser.add_argument('--bs', default=32)
    arg_parser.add_argument('--dropout', default=0.2)
    arg_parser.add_argument('--hidden_layer_size', default=64)
    arg_parser.add_argument('--epochs', default=20)

    args = arg_parser.parse_args()
    config = {'lr': 1e-3,
                'bs': 32,
                'dropout': 0.2,
                'hidden_layer_size': 64,
                'epochs': 20,
                'run_test': args.run_test,
                }

    if not args.run_test:
        wb.login()

        wb.init(project='bird_ID', config=config)
        config = wb.config
    else:
        config = dotdict(config)

    main(config)