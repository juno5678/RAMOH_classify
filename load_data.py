import xlrd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


def read_excel_file(filename):
    # 打開 Excel 檔案
    workbook = xlrd.open_workbook(filename)
    # 選取第一個工作表
    sheet = workbook.sheet_by_index(0)

    # 讀取第一欄和第二、三欄的數值
    column1 = []
    column2_3 = []
    for row_num in range(1, sheet.nrows):
        cell_value1 = sheet.cell_value(row_num, 0)  # 第一欄的數值
        cell_value2_3 = [sheet.cell_value(row_num, col)
                         for col in range(1, 3)]  # 第二、三欄的數值
        column1.append(cell_value1)
        column2_3.append(cell_value2_3)

    column2_3 = np.array(column2_3)

    return column1, column2_3

# 資料生成器類別


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, preprocess_fn):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn

        self.data_augmenter = ImageDataGenerator(
            rotation_range=20, horizontal_flip=True)
        # self.data_augmenter = ImageDataGenerator(
        #    rotation_range=20,
        #    width_shift_range=0.1,
        #    height_shift_range=0.1,
        #    zoom_range=0.2,
        #    horizontal_flip=True,
        #    brightness_range=(0.8, 1.2)
        # )

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def __getitem__(self, index):
        batch_video_paths = self.video_paths[index *
                                             self.batch_size: (index+1)*self.batch_size]
        batch_labels = self.labels[index *
                                   self.batch_size: (index+1)*self.batch_size]

        batch_video_sequences = []
        for video_path in batch_video_paths:
            print(video_path)
            count = 0
            capture = cv2.VideoCapture(video_path)
            frames = []

            seed = np.random.randint(0, 10000)

            while capture.isOpened():
                count += 1
                ret, frame = capture.read()
                if not ret or count >= 600:
                    break

                # 對每個影格進行預處理
                # augmented_frame = self.data_augmenter.random_transform(
                #    frame.astype(np.float32), seed=seed)
                #processed_frame = self.preprocess_fn(augmented_frame)
                processed_frame = self.preprocess_fn(frame)
                frames.append(processed_frame)
                #print("count : ", count)

            capture.release()

            batch_video_sequences.append(frames)
            print("index : ", index)
        return np.array(batch_video_sequences), np.array(batch_labels)


if __name__ == "__main__":
    # 指定 Excel 檔案的路徑
    excel_file = "../dataset/new_dataset/RAMOH_mds_updrs_label_20230516update_finish.xls"

    # 讀取 Excel 檔案
    result = read_excel_file(excel_file)

    # 印出結果
    print("第一行的字串:", result[0])
    print("第二、三行的數值 (1,1):", result[1])
