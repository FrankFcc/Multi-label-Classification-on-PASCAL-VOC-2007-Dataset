from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import random
import math

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def create_data(data_path, mode):
    allColumn_lists = []
    if mode == 0:
        dataset = '_trainval.txt'
    else:
        dataset = '_test.txt'

    for item in VOC_CLASSES:
        f = open(data_path + item + dataset, 'r')
        item_list = f.read().splitlines()
        f.close()
        item_list = [e for e in item_list if ('-' not in e)]
        column_list = []
        for e in item_list:
            e = e.split(" ", 1)[0]
            column_list.append(e)
        allColumn_lists.append(column_list)
    merged = []
    print('Number of Samples:\n-------------------------')
    for n in range(len(allColumn_lists)):
        print(VOC_CLASSES[n] + ' = ' + str(len(allColumn_lists[n])))
        merged = merged + allColumn_lists[n]

    merged_unique = list(set(merged))

    print('Total number of Unique samples = ' + str(len(merged_unique)))
    merged_unique = random.sample(merged_unique, len(merged_unique))

    all_bin_columns = []
    match_found = 0

    for column in allColumn_lists:
        bin_column = []
        for merged_file in merged_unique:
            for column_file in column:
                if (merged_file == column_file):  match_found = 1

            if (match_found == 1):
                bin_column.append(1)
                match_found = 0
            else:
                bin_column.append(0)
        all_bin_columns.append(bin_column)

    fileNames = []
    for m in merged_unique:
        m = m + '.jpg'
        fileNames.append(m)

    output_data = pd.DataFrame(
        {'Filenames': fileNames
         })
    for n in range(len(VOC_CLASSES)):
        output_data[VOC_CLASSES[n]] = all_bin_columns[n]

    pd.set_option('display.max_columns', None)

    return output_data


def generate_dataset(train_data, test_data):
    datagen = ImageDataGenerator(rescale=1. / 255.)
    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    print('For Training:')
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data[:-200],
        directory="C:/Users/44750/PycharmProjects/CSCI 1430 Test/CSCI-1430-FinalProject/data/VOCdevkit_2007/VOC2007/JPEGImages",
        x_col="Filenames",
        y_col=VOC_CLASSES,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(224, 224))

    print('For Validation:')
    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=train_data[-200:],
        directory="C:/Users/44750/PycharmProjects/CSCI 1430 Test/CSCI-1430-FinalProject/data/VOCdevkit_2007/VOC2007/JPEGImages",
        x_col="Filenames",
        y_col=VOC_CLASSES,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(224, 224))

    print('For Testing:')
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory="C:/Users/44750/PycharmProjects/CSCI 1430 Test/CSCI-1430-FinalProject/data/VOCdevkit_2007/VOC2007test/JPEGImages",
        x_col="Filenames",
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(224, 224))

    return train_generator, valid_generator, test_generator


# def preprocess_fn(img):
#     """ Preprocess function for ImageDataGenerator. """
#     img = img / 255.
#     img = self.standardize(img)
#     return img

if __name__ == '__main__':
    train_generator, valid_generator, test_generator = generate_dataset()


    print(train_generator)
