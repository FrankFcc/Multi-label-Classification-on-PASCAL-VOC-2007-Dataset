import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import pandas as pd
import preprocess
from preprocess import generate_dataset, VOC_CLASSES
from models import YourModel, VGGModel, ResNetModel
from keras import regularizers, optimizers
from matplotlib import pyplot as plt
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
import numpy as np
tf.config.run_functions_eagerly(True)

import tensorflow as tf
from keras import backend as K

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3', '5'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, 5)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=100,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


if __name__ == '__main__':
    ARGS = parse_args()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    train_generator, valid_generator, test_generator = preprocess.generate_dataset()

    os.chdir(sys.path[0])

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(224, 224, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
                          "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
                    os.sep + timestamp + os.sep
        model.summary()
    elif ARGS.task == '3':
        model = VGGModel()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
    else:
        model = ResNetModel()
        checkpoint_path = "checkpoints" + os.sep + \
                          "resnet50_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "resnet50_model" + \
                    os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.resnetv2.summary()
        model.head.summary()



    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # Print summary of model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
        loss="binary_crossentropy",
        metrics=["accuracy"])
    # model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss="binary_crossentropy", metrics=["accuracy"])


    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(checkpoint_path, ARGS.task, 5)
    ]

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    print('\ntraining\n-------------------------')
    model.fit(train_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=valid_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=100,
              callbacks=callback_list,
              initial_epoch=0,)
    print('\ntesting\n-------------------------')
    test_generator.reset()
    pred_two = model.predict(test_generator,
                                       steps=STEP_SIZE_TEST,
                                       # steps = 100,
                                       verbose=1)
    pred_bool = (pred_two > 0.5)

    predictions = pred_bool.astype(int)
    # columns should be the same order of y_col
    results = pd.DataFrame(predictions, columns=VOC_CLASSES)
    results["Filenames"] = test_generator.filenames
    ordered_cols = ["Filenames"] + VOC_CLASSES
    results = results[ordered_cols]
    results.to_csv("output.csv", encoding='utf-8', index=False)
