import tensorflow as tf
import random
import argparse
from birdsong_recognition.preprocess import *


def main(args):
    seed = 42
    random.seed(seed)

    run_test = args.run_test
    running_in_colab = args.colab

    ebirds = ['norcar', 'blujay', 'bkcchi']
    train_ds, val_ds = preprocess(running_in_colab, ebirds)
    
    """### simple model"""

    import tensorflow.keras as keras
    from tensorflow.keras import layers, losses, optimizers
    import wandb as wb
    from wandb.keras import WandbCallback

    wb.login()

    wb.init(project='bird_ID', config={'lr': 1e-3, 'bs': 32})
    config = wb.config

    keras.backend.clear_session()

    model = tf.keras.Sequential([
                layers.Conv2D(filters=32, kernel_size=(4,4), strides=1, activation='relu', input_shape=(284, 257, 1)),
                layers.MaxPool2D(pool_size=(4,4)),
                layers.Conv2D(filters=64, kernel_size=(4,4), strides=1, activation='relu'),
                layers.MaxPool2D(pool_size=(4,4)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(3)
    ])

    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_ = train_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)
    val_ds_ = val_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)

    print('*'*30)
    print('Starting training')
    print('*'*30)

    if run_test:
        model.fit(train_ds_, epochs=2, validation_data=val_ds_, steps_per_epoch=2)
    else:
        model.fit(train_ds_, epochs=2, validation_data=val_ds_, callbacks=[WandbCallback()])       

    wb.finish()

if __name__== '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_test', action='store_true')
    arg_parser.add_argument('--colab', action='store_true')
    args = arg_parser.parse_args()
    main(args)