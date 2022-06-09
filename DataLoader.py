import Config
import os
import pandas as pd
import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from PIL import Image
from imutils import paths
import pathlib
import matplotlib.pyplot as plt

class dataProcess:
    def combinImg(mode,dataList):
        for i in range(len(dataList)):
            img_F = cv2.imread(os.path.join(Config.PATH_DATA_ALL,dataList[i]+'_F.jpg'))
            img_L = cv2.imread(os.path.join(Config.PATH_DATA_ALL,dataList[i]+'_L.jpg'))
            img_R = cv2.imread(os.path.join(Config.PATH_DATA_ALL,dataList[i]+'_R.jpg'))
            if img_F.shape==img_L.shape and img_F.shape==img_R.shape:
                if mode=='basic':
                    res = np.hstack((img_L,img_F,img_R))
                    res = cv2.imwrite(os.path.join(Config.PATH_DATA_ALL_COMBIN,dataList[i]+'.jpg'),res,[cv2.IMWRITE_JPEG_QUALITY, 100])
                else:
                    print('1')
            else:
                img_L=cv2.resize(img_L,(img_F.shape[1],img_F.shape[0]))
                img_R=cv2.resize(img_R,(img_F.shape[1],img_F.shape[0]))
                if mode=='basic':
                    res = np.hstack((img_L,img_F,img_R))
                    res = cv2.imwrite(os.path.join(Config.PATH_DATA_ALL_COMBIN,dataList[i]+'.jpg'),res,[cv2.IMWRITE_JPEG_QUALITY, 100])
                else:
                    print('1')
    def get_image(dataList):
        data=[]
        for i in range(len(dataList)):
            image = load_img(
                os.path.join(Config.PATH_DATA_ALL_COMBIN,dataList[i]+'.jpg'),
                grayscale=True,
                #color_mode='rgb',
                target_size=(224,224)
            )
            image = img_to_array(image,data_format=None)
            image = preprocess_input(image)
            data.append(image)
        # transfer to numpy array
        data = np.array(data,dtype='float32')
        return data
    def get_label_weight(labelList):
        label_num = np.bincount(labelList)
    def get_List_data_label(path_csv):
        df = pd.read_csv(path_csv,encoding='UTF-8')
        print(df)
        dataList = []
        labelList = []
        for i in range(len(df['Label'])-1):
            dataList.append(df['File_Index'][i])
            labelList.append(int(df['Label_Avg_round'][i]))
        print(dataList) 
        print(labelList)
        return dataList,labelList

#class dataPreprocess:
class dataLoader:
    def TrainTest(data,labels):
        (train_x, test_x, train_y, test_y) = train_test_split(
            data,
            labels,
            test_size=0.10, 
            stratify=labels, 
            random_state=None, 
            shuffle=True)
        (train_x, valid_x, train_y, valid_y) = train_test_split(
            train_x,
            train_y,
            test_size=0.2, 
            stratify=train_y, 
            random_state=None, 
            shuffle=True)
        return train_x,train_y,valid_x,valid_y,test_x,test_y
    
    def loading_byDir(path_Data_dir):
        ImagePaths = list(paths.list_images(path_Data_dir))
        print(path_Data_dir)
        data=[]
        labels=[]#→根據資料夾
        for imagepath in ImagePaths:
            label=imagepath.split(os.path.sep)[-2]
            label=os.path.split(label)[1]
            labels.append(label)
            #print(labels)
            image = load_img(
                imagepath, 
                grayscale=False,
                color_mode='rgb',
                target_size=(224,224))
            #print(image)
            image = img_to_array(image,data_format=None)
            image = preprocess_input(image)
            data.append(image)
        #print(labels,'\n')
        print("[INFO] all data...",len(data))
        # transfer to numpy
        data = np.array(data, dtype="float32")
        #data = data.astype('float32')/255
        labels = np.array(labels)
        labelsarr =labels.copy()
        #print(labels)
        # one-hot
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        #print(labels)
        #print('label list ',list(lb.classes_))
        labels = to_categorical(labels)
        print(labels)
        print('[DATA] class name: ',list(lb.classes_))# ['Mixed', 'dry', 'net', 'oil']
        return data,labels,labelsarr

class dataLoader_byDIR:
    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show()
    def parse_image(filename):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image,channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        return image, label
    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        image = tf.io.read_file(file_path)
        print(label)
        return image, label
    def main(data_path):
        dirroot = pathlib.Path(data_path)
        for item in dirroot.glob('*'):
            print(item)
        list_ds = tf.data.Dataset.list_files(str(dirroot/'*/*'))
        for f in list_ds.take(5):
            print(f.numpy())
        
        
        labeled_ds = list_ds.map(dataLoader_byDIR.process_path)
        for image_raw, label_text in labeled_ds.take(1):
            print(repr(image_raw.numpy()[:100]))
            print()
            print(label_text.numpy())
        #file_path = next(iter(list_ds))
        #image, label = dataLoader_byDIR.parse_image(file_path)
        
        data_ds = list_ds.map(dataLoader_byDIR.parse_image)

        print(data_ds)

        
class dataAugment:
    def main(images,label):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            #iaa.Invert(0.5),
            #sometimes(iaa.Cutout(nb_iterations=2)),
            sometimes(iaa.Jigsaw(nb_rows=(10),nb_cols=(10))),
            sometimes(iaa.Jigsaw(nb_rows=(1,4),nb_cols=(1,4))),
            #sometimes(iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))),
            sometimes(iaa.Fliplr(0.5)),
            sometimes(iaa.Flipud(0.5)),
            #iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            #iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
            sometimes(iaa.Rotate((-45,180))),
            sometimes(iaa.Affine(
                translate_px={"x": 15, "y": 15},
                scale=(0.8, 0.95),
            ))  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        ])
        images_aug = seq(images=images)
        return images_aug,label
class AcneECK:
    def byCSV():
        dataList,labelList=dataProcess.get_List_data_label(Config.PATH_CSV)
        data = dataProcess.get_image(dataList)
        print(data.shape)
        return data,labelList
    def preprocess(images, labels):
        scale_layer = tf.keras.layers.Rescaling(1./255)
        images = tf.image.resize(scale_layer(images),[120, 120], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return images, labels
    def byDir2():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            Config.PATH_DATA_TRAIN,
            validation_split=0.2,
            subset="train",
            seed=None,
            image_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
            batch_size=Config.BS)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            Config.PATH_DATA_TEST,
            validation_split=None,
            subset="test",
            seed=None,
            image_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
            batch_size=Config.BS)
        train_ds = train_ds.map(AcneECK.preprocess)
        print(train_ds.shape)
        print(test_ds.shape)
        return train_ds, test_ds
    def byDir():
        train_x,train_y,labelsarr2 = dataLoader.loading_byDir(Config.PATH_DATA_TRAIN)
        test_x,test_y,labelsarr = dataLoader.loading_byDir(Config.PATH_DATA_TEST)
        train_x,valid_x,train_y,valid_y = train_test_split(
            train_x,
            train_y,
            test_size=0.10, 
            stratify=train_y, 
            random_state=None, 
            shuffle=True)
        return train_x,train_y,valid_x,valid_y,test_x,test_y

