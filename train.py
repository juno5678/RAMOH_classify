import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_data import read_excel_file
from load_data import DataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Dao's rPPG estimate heart rate")
    parser.add_argument(
        "-g", "--GT_path", type=str, required=True, help="path of Ground Truth excel"
    )
    parser.add_argument(
        "-r", "--root_dir_path", type=str, required=True, help="path of directory with video sample "
    )
    parser.add_argument(
        "--output_dir",
        default="./result/",
        type=str,
        help="the path of result (default : %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--display",
        action='store_true',
        default=False,
        help="display training result or not (default : %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def augment_data(image, label):
    augmented_images = datagen.flow(image, shuffle=False).next()
    return augmented_images, label

# 影片預處理函數


def preprocess_frame(frame):
    # size = (960, 540)
    size = (112, 112)
    # 在這裡執行你的預處理操作，例如調整大小、正規化等
    frame = cv2.resize(frame, size)
    processed_frame = frame / 255.0  # 此處示例將影格進行正規化，範圍縮放到[0, 1]
    return processed_frame


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # dataset path
    # dataset_file = "../dataset/new_dataset/RAMOH_mds_updrs_label_20230516update_finish.xls"
    dataset_file = args.GT_path
    videoName, labels = read_excel_file(dataset_file)
    # 讀取影片並處理
    # root_directory = "../dataset/new_dataset/video/"
    root_directory = args.root_dir_path
    video_paths = [os.path.join(root_directory, video) for video in videoName]
    height, width, channels = 112, 112, 3
    # 切分訓練、驗證和測試集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.2)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.25)

    batch_size = 1
    train_generator = DataGenerator(
        train_paths, train_labels, batch_size, preprocess_frame)

    val_generator = DataGenerator(
        val_paths, val_labels, batch_size, preprocess_frame)

    # 定義神經網路模型
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu',
               input_shape=(600, height, width, channels)),
        MaxPooling3D((2, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Conv3D(128, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Conv3D(256, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Conv3D(512, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        # Dense(128, activation='relu'),
        Dropout(0.5),
        # Dense(64, activation='relu'),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(2, activation='relu')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae', rmse])
    print(model.summary())

    early_stopping = EarlyStopping(
        patience=3, monitor='val_loss', restore_best_weights=True)

    save_path = os.path.join(args.output_dir, 'best_model.h5')
    checkpoint = ModelCheckpoint(
        save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# 訓練模型
    epochs = 30
    steps_per_epoch = len(train_paths) // batch_size
    print("steps per epoch : ", steps_per_epoch)
    print("train paths : ", len(train_paths))
    validation_steps = len(val_paths) // batch_size
    print("validation steps : ", validation_steps)
    print("train paths : ", len(val_paths))

    train_start_time = time.time()
    history = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        callbacks=[early_stopping, checkpoint])
    train_end_time = time.time()

    # 獲取訓練和驗證損失
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 獲取訓練和驗證準確度
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    if args.display:
        # 繪製訓練和驗證損失
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 繪製訓練和驗證準確度
        plt.plot(history.history['mae'], label='Training Accuracy')
        plt.plot(history.history['val_mae'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    # 評估模型
    test_generator = DataGenerator(
        test_paths, test_labels, batch_size, preprocess_frame)
    test_steps = len(test_paths) // batch_size

    model = load_model(save_path, custom_objects={'rmse': rmse})

    test_start_time = time.time()
    test_loss, test_mae, test_rmse = model.evaluate(
        test_generator, steps=test_steps)
    test_end_time = time.time()

    train_time = train_end_time - train_start_time
    test_time = test_end_time - test_start_time
    one_test_time = test_time / len(test_paths)

    print("test dataset loss:", test_loss)
    print("test dataset mae:", test_mae)
    print("test dataset rmse:", test_rmse)
    print("training time: ", train_time)
    print("test time for all test dataset: ", test_time)
    print("test time for one sample: ", one_test_time)
