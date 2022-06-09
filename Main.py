from random import sample
import Config
import DataLoader
import ModelLoader
import Visualization
import Devices
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# --------------------#
# GPU
# --------------------#
def devices():
    Devices.GPU()
# --------------------#
# Data
# --------------------#
def load_data():
    train_x,train_y,valid_x,valid_y,test_x,test_y = DataLoader.AcneECK.byDir()
    return train_x,train_y,valid_x,valid_y,test_x,test_y

def load_data2():
    train_ds,val_ds = DataLoader.AcneECK.byDir2()
    return train_ds,val_ds

def load_data3():
    (train_x,train_y) = DataLoader.dataLoader_byDIR.main(Config.PATH_DATA_TRAIN)
    (test_x,test_y) = DataLoader.dataLoader_byDIR.main(Config.PATH_DATA_TEST)
    train_x = train_x.reshape(460, 4).astype("float32") / 255
    test_x = test_x.reshape(10, 4).astype("float32") / 255


    return (train_x,train_y),(test_x,test_y)

def explore(train_x,train_y,valid_x,valid_y,test_x,test_y):
    print('train set: ',train_x.shape,train_y.shape)
    print('valid set: ',valid_x.shape,valid_y.shape)
    print('test set: ',test_x.shape,test_y.shape)
    '''
    plt.figure()
    plt.imshow(train_x[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    '''

def normalize(x,y):
    x = tf.image.per_image_standardization(x)
    return x,y

def augmentation(x,y):
    x = tf.image.random_flip_left_right(x)
    return x, y



# --------------------#
# Model
# --------------------#
def build_model():
    model = ModelLoader.build_model()
    #model.summary()
    return model
    
#----------------------------------------------------------#
# TRAIN Model
#----------------------------------------------------------#
# [Tensorboard]
# -run this in your cmd
# --$ tensorboard --logdir=./logs
# [CALLBACKS]
# -EarlyStopping(monitor='val_accuracy',min_delta = 0.0001,patience=5,verbose=True)
# --In 5 patience, the val_accuray doesn't improve in range 0.0001
# [CSVLogger]
#----------------------------------------------------------#
def train_model(model,train_x,train_y,valid_x,valid_y):
    aug = ImageDataGenerator(
        #my_preprocessing.filter.filter_processing
        #seq
        #Enhance_Image.claHE,
        #rotation_range=20,
        #zoom_range=0.15,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.15,
        #horizontal_flip=True,
        #fill_mode="nearest"
        )

    class dataAugment:
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
        aug_train = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False,
            featurewise_std_normalization=False, 
            samplewise_std_normalization=False,
            zca_whitening=False, zca_epsilon=1e-06, 
            rotation_range=130, 
            width_shift_range=0.0,
            height_shift_range=0.0, 
            brightness_range=None, 
            shear_range=0.0, 
            zoom_range=0.0,
            channel_shift_range=0.0, 
            fill_mode='nearest', 
            cval=0.0,
            horizontal_flip=True, 
            vertical_flip=True, 
            rescale=None,
            preprocessing_function=None, 
            data_format=None, 
            validation_split=0.0, 
            dtype=None
        )
    print('#--------Traing Start-------#')
    def LR_scheduler(epoch):
        if epoch < 100:
            return Config.INIT_LR
        else:
            return Config.INIT_LR * tf.math.exp(-0.1)
    CALLBACKS=[
        #tf.keras.callbacks.TensorBoard(log_dir=Config.path_log_dir,histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=Config.PATH_CHECKPOINT,save_weights_only=True,monitor='val_accuracy',mode='max',save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(LR_scheduler)
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.8,patience=5,verbose=0,mode='auto',baseline=None,restore_best_weights=False)
    ]
    #train_x,train_y = augmentation(train_x,train_y)
    H = model.fit(
        #dataAugment.aug_train.flow(train_x, train_y, batch_size=Config.BS),
        x=train_x,
        y=train_y,
        batch_size=Config.BS,
        epochs=Config.EPOCHS,
        verbose = 1,
        callbacks=CALLBACKS,
        validation_split=0.0,
        #validation_data=dataAugment.aug_train.flow(valid_x, valid_y,batch_size=8),
        validation_data = (valid_x,valid_y),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=len(train_x) // Config.BS,        
        validation_steps=len(valid_x) // Config.BS,
        validation_batch_size = None,
        validation_freq = 1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False     
        )
    print('#--------Traing Done-------#')
    return H,model

def evaluate_model(model,train_x,train_y,test_x,test_y):
    #----------------------------------------------------------#
    # EVALUATE Model
    #----------------------------------------------------------#
    # 1.模型的BatchNormalization，Dropout，LayerNormalization等优化手段只在fit时，对训练集有用;
    # 2.在进行evaluate()的时候，这些优化都会失效，因此，再次进行evaluate(x_train,y_train),就算添加了batchsize，也不能达到相同的评估计算结果。
    # ————————————————
    # 版权声明：本文为CSDN博主「风筝不是风」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/weixin_45279187/article/details/110194739
    print("[INFO] EVALUATE Model...")
    print("[INFO] EVALUATE on Train set...")
    try:
        eval_loss_trainset,eval_accuracy_trainset,*is_anything_else_being_returned  = model.evaluate(
            x=train_x, 
            y=train_y,
            batch_size=32,
            verbose=1,
            sample_weight=None,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            return_dict=False,
            )
        print('[OUTPUT] eval_loss.. ',eval_loss_trainset)
        print('[OUTPUT] eval_accuracy.. ',eval_accuracy_trainset)
        try:
            print('[OUTPUT] *is_anything_else_being_returned.. ',*is_anything_else_being_returned)
        except:
            print('[ERROR] Evaluate error2')
    except:
        print('[ERROR]:ERROR: Evaluate error')
    print("[INFO] EVALUATE on Test set...")
    try:
        eval_loss_testset,eval_accuracy_testset,*is_anything_else_being_returned  = model.evaluate(
            x=test_x, y=test_y, batch_size=32, verbose=1, sample_weight=None, steps=None,
            callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
            return_dict=False,
            )
        print('[OUTPUT] eval_loss.. ',eval_loss_testset)
        print('[OUTPUT] eval_accuracy.. ',eval_accuracy_testset)
        try:
            print('[OUTPUT] *is_anything_else_being_returned.. ',*is_anything_else_being_returned)
        except:
            print('[ERROR] Evaluate error2')
    except:
        print('[ERROR]:ERROR: Evaluate error')
    
    return eval_loss_trainset,eval_accuracy_trainset,eval_loss_testset,eval_accuracy_testset

def predict_model(model,test_x):
    #----------------------------------------------------------#
    # PREDICT Model
    #----------------------------------------------------------#
    print("[INFO]:RUN: PREDICT Model...")
    try:
        #predIdxs = model.predict(myData.test_x, batch_size=myconfig.BS)
        predictions = model.predict(
            test_x, 
            batch_size=None, 
            verbose=0, 
            steps=None, 
            callbacks=None, 
            max_queue_size=None,
            workers=1, 
            use_multiprocessing=False
        )
        print('#--------prediction-------#')
        #print(predictions)
        print('#--------prediction.shape-------#')
        print(predictions.shape)
        print('#--------classification-------#')
        classification = np.argmax(predictions,axis=1)
        print(classification)
        print('#--------confusion_matrix-------#')
        from sklearn.metrics import confusion_matrix
        test_y = np.argmax(test_y,axis=1)
        CM = confusion_matrix(
            test_y, classification
        )
        print(CM)
    except:
        print('[INFO]:ERROR: Predict fail!')

    try:
        for i in range(len(test_x)):
            plt.figure()
            plt.subplot(1,2,1)
            Visualization.plot_predictions.plot_image(i,predictions[i],test_y,test_x)
            plt.subplot(1,2,2)
            Visualization.plot_predictions.plot_value_array(i,predictions[i],test_y)
            plt.savefig(
                Config.PATH_PREDICTIONS+str(i)+'.jpg'
            )
    except:
        print('[INFO]:ERROR: Predict Fail!')

    
def save_model(model,eval_accuracy,eval_loss):
    #----------------------------------------------------------#
    # SAVING Model
    #----------------------------------------------------------#
    '''time getting'''
    import time
    time_now = time.localtime(time.time())
    time_save = str(time_now[0])+'_'+str(time_now[1])+str(time_now[2])+'_'+str(time_now[3])+str(time_now[4])

    print("[INFO] SAVING Model...")
    path_save_model = Config.PATH_MODEL+'[EPOCHS_'+str(Config.EPOCHS)+']'+'[ACC_'+str(round(eval_accuracy,4))+']'+'[LOSS_'+str(round(eval_loss,4))+'].'+Config.MODEL_FORMAT
    print('path_save_model: ',path_save_model)
    #model.save('./model_save/'+str(myData.name_model_save_dataset)+'_'+str(myconfig.backbone)+'_'+str(time_save)+'_'+str(myconfig.EPOCHS)+'_'+str(eval_accuracy)+'.'+myconfig.modelFormat, save_format=myconfig.modelFormat)
    model.save(
        path_save_model,
        #overwrite=True,
        #include_optimizer=True,
        save_format=Config.MODEL_FORMAT)




devices()
train_x,train_y,valid_x,valid_y,test_x,test_y = load_data()
#explore(train_x,train_y,valid_x,valid_y,test_x,test_y)
#train_dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
#train_dataset = (train_dataset.map(augmentation))
#print(train_dataset)



model = build_model()
H,model = train_model(model,train_x,train_y,valid_x,valid_y)
eval_loss_trainset,eval_accuracy_trainset,eval_loss_testset,eval_accuracy_testset = evaluate_model(model,train_x,train_y,test_x,test_y)
save_model(model,eval_accuracy_testset,eval_loss_testset)
Visualization.plot_trainning_history.plot(H)