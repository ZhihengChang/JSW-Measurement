import os

import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , BatchNormalization
"""
This script provides model compiling, training, and saving.

NOTE: 
VGG can be any other model just by removing the activation function at the output layer
and change the output number at the end to predict numeric JSW measurement
Also set the loss function to MSE
"""


def vgg_19(input_shape, output, alpha=1e-04):
    """
    VGG_19 model from scratch
    This is NOT a typical classification model, the final output layer does not have an activation function
    Therefore it is outputting a numeric value for JSW measurement
    The model is compile based on the MSE
    NOTE: Could potentially import pretrained model from keras without weights and head
    :param input_shape: the x data (full joint or subregion images) shape
    :param output: number of numeric output: 1: predicting JSW_ave; 5: predicting JSW1-JSW5 for the subregion
        NOTE: The y_train and y_val has to be match with the ouput as well
    :param alpha: The learning rate, default is 0.0001
    :return: The complied model
    """
    print("creating VGG-19 model with input shape {}...".format(input_shape))
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # No point of using dropout since there is BatchNormalization
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output))

    print("compling model...")
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=alpha),
        metrics=['mse']
    )

    print("done.\n")
    return model


def train_save(model, name, x_train, y_train, x_val, y_val, batch_size, epochs):
    """
    Train and save the given model (and its training history) using the given data and parameters
    model save path is '../saved_model/' change in method if needed
    history save path is '../history/' change in method if needed
    :param model: Deep learning model
    :param name: model name. This name will be used when saving both model and its training history
    :param x_train: training data: joint images
    :param y_train: training label: JSW measurements
    :param x_val: validation data
    :param y_val: validation label
    :param batch_size: batch size
    :param epochs: number of epochs to train
    :return: none
    """
    print("start training the model...")
    # early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')

    # Training the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        # callbacks=[early]
    )

    # Save model and its history
    path = os.path.dirname(__file__)
    model.save(os.path.join(path, '../saved_model/' + name))

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(path, '../history/' + name + '_history.csv'), index=False)
