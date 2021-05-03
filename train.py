import tensorflow as tf
import random
from birdsong_recognition.preprocess import *


def main():
    seed = 42
    random.seed(seed)

    running_in_colab = False

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

    #LargeDataset
    model.fit(train_ds_, epochs=2, validation_data=val_ds_, callbacks=[WandbCallback()], steps_per_epoch=2)

    wb.finish()

if __name__== '__main__':
    main()