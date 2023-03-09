import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.image as image
import tensorflow.random as random
import tensorflow.keras.callbacks as callbacks
from tensorflow.train import Checkpoint
import tensorflow.data as data
import pickle
import tensorflow as tf

# phys = tf.config.list_physical_devices('GPU')
# print("\n\n", phys[0])
# tf.config.set_logical_device_configuration(
#     phys[0],
#     [
#             tf.config.LogicalDeviceConfiguration(memory_limit=10000),
#     ]
# )
# print(tf.config.list_logical_devices('GPU'), "\n\n")
        
train = utils.image_dataset_from_directory(
        "images",
        label_mode = 'categorical',
        batch_size = 64,
        image_size = (320,258),
        seed = 23,
        validation_split = 0.3,
        subset = "training",
)
f = open("./fruit_model_save/class_names.data", "wb")
pickle.dump(train.class_names, f)
f.close()

test = utils.image_dataset_from_directory(
        "images",
        label_mode = 'categorical',
        batch_size = 64,
        image_size = (320,258),
        seed = 23,
        validation_split = 0.3,
        subset = "validation",

)
train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = data.AUTOTUNE)

# train_brightness = train.map(lambda x, y: (image.stateless_random_brightness(x, 0.5, (2,3)), y))
# train_contrast = train.map(lambda x, y: (image.stateless_random_contrast(x, 0.2, 0.5 ,(2,3)), y))
# train_flip_left = train.map(lambda x, y: (image.stateless_random_flip_left_right(x,(2,3)), y))
# train_flip_up = train.map(lambda x, y: (image.stateless_random_flip_up_down(x,(2,3)), y))
# train_hue = train.map(lambda x, y: (image.stateless_random_hue(x, 0.3 ,(2,3)), y))
# train_saturation = train.map(lambda x, y: (image.stateless_random_saturation(x, 0.25, 1 ,(2,3)), y))

# train = train.concatenate(train_saturation)
# train = train.concatenate(train_contrast)
# train = train.concatenate(train_brightness)
# train = train.concatenate(train_flip_left)
# train = train.concatenate(train_flip_up)
# train = train.concatenate(train_hue)

class Net():
        def __init__(self, input_shape):
                self.model = models.Sequential()
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0),(1,0)),
                        input_shape = input_shape))
                self.model.add(layers.RandomContrast(factor = 0.5))
                self.model.add(layers.RandomRotation(factor = 0.5))
                self.model.add(layers.RandomZoom(height_factor = 0.5))
                self.model.add(layers.Conv2D(
                        15, # filters
                        (6,5), # size
                        strides = (3,3), #step size
                        activation = 'relu',
                )) #output 105x82x15
                self.model.add(layers.Conv2D(
                        15,
                        (5,4),
                        strides = (2,2),
                        activation = 'relu'
                )) #50X39X15
                self.model.add(layers.Dropout(0.3))
                self.model.add(layers.DepthwiseConv2D(
                3, # size
                strides = (1,1), #step size
                depth_multiplier = 2,
                activation = 'relu',
                )) #47x36x15
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0), (0,0))
                ))
                #output =48x36x 15
                self.model.add(layers.MaxPool2D(pool_size=2))
                #output 24x18x 15 
                self.model.add(layers.DepthwiseConv2D(
                       3,
                        strides = (1,1),
                        activation = 'relu'
                ))
                #21x15x15
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0), (1,0))
                ))
                #22x16x15
                self.model.add(layers.MaxPool2D(pool_size=2))
                #11x8x15=1320
                self.model.add(layers.Flatten())
                #output: 2016
                self.model.add(layers.Dense(512, activation = 'relu'))
                self.model.add(layers.Dense(128, activation = 'relu'))
                self.model.add(layers.Dense(32, activation = 'relu'))
                self.model.add(layers.Dense(15, activation = 'softmax'))
                self.loss = losses.CategoricalCrossentropy()
                self.optimizer = optimizers.SGD(learning_rate = 0.0001, momentum=0.5, decay = 0.0001/200)
                self.model.compile(
                        loss = self.loss,
                        optimizer = self.optimizer,
                        metrics = ["accuracy"]
                )
        def __str__(self):
                self.model.summary()
                return ""

net =  Net((320,258,3))
checkpoint = Checkpoint(net.model)
# checkpoint.restore('checkpoints/checkpoints_20')
callbacks = [
    callbacks.ModelCheckpoint(
        'checkpoints/checkpoints_{epoch:02d}', 
        verbose = 2, 
        save_freq = 76,
    )
]
print(net)
net.model.fit(
        train,
        batch_size = 64,
        epochs = 200,
        verbose  = 2,
        validation_data = test,
        validation_batch_size = 64,
        callbacks=callbacks,
        initial_epoch = 0,
)

net.model.save("fruit_model_save") 