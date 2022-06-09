from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import datetime
import time
import os
#----------------------------------------------------------#
# System
#----------------------------------------------------------#
SYSTEM_TIME = datetime.datetime.now().strftime('%Y%m%d-%H%M')
TIME_NOW = time.localtime(time.time())
#----------------------------------------------------------#
# Data
#----------------------------------------------------------#
CLASS_NAME=['LEVEL_0,LEVEL_1,LEVEL_2,LEVEL_3']
DATASET='AcneECK'
CLASSES=7
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_CHANNEL=3
INPUT_SHAPE=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL)
PROPORTION_TRAIN_TEST=0.2
PROPORTION_TRAIN_VALIDATION=0.1
PATH_DATA_ALL = r'F:/#DataSet/#Acne/AcneECK/Group_Order_index/AcneECK_ALL/'
PATH_DATA_ALL_COMBIN = r'F:/#DataSet/#Acne/AcneECK/Group_Order_index/AcneECK_ALL_Combin/'
PATH_DATA_TRAIN = r'F:/#DataSet/#Acne/AcneECK/Group_Order_to_Class/Train/'
PATH_DATA_TEST = r'F:/#DataSet/#Acne/AcneECK/Group_Order_to_Class/Test/'
PATH_CSV=r'F:/#DataSet/#Acne/AcneECK/annoaction_2022_0609.csv'
#----------------------------------------------------------#
# Model
#----------------------------------------------------------#
BACKBONE='Efficient'
MODEL_FORMAT='h5'
PRETRAIN_MODEL_WEIGHT='imagenet'
INCLUDE_TOP = False
DRAW_PLOT = False
OUTPUT_MODEL = 'R'#Regression
#----------------------------------------------------------#
# Model Phase
#----------------------------------------------------------#
MODEL_PHASE_TRAIN=True
MODEL_PHASE_EVALUATE=True
INIT_LR = 1e-4
EPOCHS = 2
BS = 8 
OPT = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
WEIGHT=r'E:/Project/FER/logs/checkpoints/epoch41-[0.8453].h5'
METRICS = [
    #tf.keras.metrics.Accuracy(name ='accuracy'),
    #tf.keras.metrics.BinaryAccuracy(name='binaryaccuracy'),
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),#specificity
    tf.keras.metrics.Recall(name='recall'),#Sensitivity
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.TruePositives(name='TP'),
    tf.keras.metrics.TrueNegatives(name='TN'),
    tf.keras.metrics.FalsePositives(name='FP'),
    tf.keras.metrics.FalseNegatives(name='FN'),
    tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
]

#----------------------------------------------------------#
# Saving
#----------------------------------------------------------#
PATH_CHECKPOINT = './logs/checkpoints/epoch.{epoch:02d}-{val_accuracy:.4f}.h5'
PATH_PRETRAINMODEL=r'E:/Project/FER/logs/checkpoints/epoch41-[0.8453].h5'
PATH_PLOT = './logs/history/'+DATASET+'_'+BACKBONE+'_'+str(EPOCHS)+'_'+SYSTEM_TIME
PATH_MODEL = os.path.join('./models',DATASET,BACKBONE,str(TIME_NOW[0]),str(TIME_NOW[1]))
PATH_PLOT_MODEL = './logs/model_plot/'+DATASET+'_'+BACKBONE+'_'+str(EPOCHS)+'_'+SYSTEM_TIME
PATH_PREDICTIONS= './predictions/'




