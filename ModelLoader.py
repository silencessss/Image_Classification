import Config
import tensorflow as tf
import tensorflow_hub as hub
import efficientnet.tfkeras as efn


data_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ],
    name="data_augmentation",
)

class get_Input_Flow:
    def Input_Flow(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = data_augmentation(inputs)
        x = tf.keras.layers.Rescaling(1.0/255.0)
        return x




class get_backbone:
    def EfficientNetB7():
        backbone = tf.keras.applications.efficientnet.EfficientNetB7(
            include_top = Config.INCLUDE_TOP,
            weights = Config.PRETRAIN_MODEL_WEIGHT,
            input_tensor = None,
            input_shape=(Config.INPUT_SHAPE),
            pooling = None,
            classes = Config.CLASSES,
            classifier_activation='softmax'
        )
        return backbone

class get_FC:
    def Fully_Connected(baseModel):
        headModel = baseModel.output
        #headModel = CoordAtt_bolck.CoordAtt(headModel,reduction = 32)
        headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
        #headModel = tf.keras.layers.MaxPooling2D(pool_size=(7,7))(headModel)
        headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        #headModel = Dense(64, activation="tanh")(headModel)#relu,tanh
        headModel = tf.keras.layers.Dense(32, activation="tanh")(headModel)#relu,tanh
        #headModel = Dropout(0.2)(headModel)
        headModel = tf.keras.layers.Dense(4, activation="softmax")(headModel)
        outModel = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)
        return outModel

def plot_model(model):
    tf.keras.utils.plot_model(
        model,
        to_file=Config.PATH_PLOT_MODEL,
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )



def build_model():
    #x = get_Input_Flow.Input_Flow(Config.INPUT_SHAPE)
    if Config.BACKBONE=='Efficient':
        backbone = get_backbone.EfficientNetB7()
        model = get_FC.Fully_Connected(backbone)
    else:
        print('please check your config!!!')

    if Config.DRAW_PLOT==True:
        plot_model(model)
    if Config.OUTPUT_MODEL == 'Regression':
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(), 
            optimizer=Config.OPT,
            metrics=Config.METRICS
            )
    else:
        model.compile(
            optimizer=Config.OPT,
            loss="categorical_crossentropy",            
            metrics=Config.METRICS,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            #jit_compile=None,
            )
    return model