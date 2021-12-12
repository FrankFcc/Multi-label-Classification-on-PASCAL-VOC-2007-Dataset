import tensorflow as tf
from tensorflow.keras.layers import \
Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2


class YourModel(tf.keras.Model):

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-6)

        self.architecture = [
            Conv2D(64, 3, 1, activation="relu", padding="same"),
            Dropout(rate=0.1),
            Conv2D(64, 3, 1, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(3, strides=2, padding="valid"),
            Conv2D(128, 3, 1, activation="relu", padding="same"),
            Dropout(rate=0.1),
            Conv2D(128, 3, 1, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(3, strides=2, padding="valid"),
            Conv2D(256, 3, 1, activation="relu", padding="same"),
            Dropout(rate=0.1),
            Conv2D(256, 3, 1, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPool2D(3, strides=2, padding="valid"),
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(rate=0.3),
            Dense(64, activation='relu'),
            Dropout(rate=0.3),
            Dense(20, activation='sigmoid')
        ]

    def call(self, x):
        for layer in self.architecture:
             x = layer(x)
        return x


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-6)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        for layer in self.vgg16:
            layer.trainable = False

        self.head = [
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(rate=0.3),
            Dense(64, activation='relu'),
            Dropout(rate=0.3),
            Dense(20, activation='sigmoid')
        ]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x


class ResNetModel(tf.keras.Model):
    def __init__(self):
        super(ResNetModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-6)

        self.resnetv2 = ResNet50V2(include_top=False, weights='imagenet')

        for layer in self.resnetv2.layers[:143]:
            layer.trainable = False

        self.head = [
            Flatten(),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(rate=0.3),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(rate=0.3),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(rate=0.3),
            BatchNormalization(),
            Dense(20, activation='sigmoid')
        ]

        # Don't change the below:
        self.resnetv2 = tf.keras.Sequential(self.resnetv2, name="resnet_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.resnetv2(x)
        x = self.head(x)

        return x