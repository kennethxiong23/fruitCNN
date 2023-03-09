from pprint import pprint

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.callbacks as callbacks
from tensorflow.data import Dataset
from tensorflow.train import Checkpoint
import tensorflow.image as image
import tensorflow as tf
import numpy as np
import cv2

train = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 64,
    image_size = (320, 258),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 64,
    image_size = (320, 258),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)


class_names = train.class_names

print("Class Names:")
pprint(class_names)



class Net():
        def __init__(self, input_shape):
                self.model = models.Sequential()
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0),(1,0)),
                        input_shape = input_shape))
                self.model.add(layers.RandomContrast(factor = 0.5))
                self.model.add(layers.RandomRotation(factor = 0.5))
                self.model.add(layers.Conv2D(
                        24, # filters
                        (15,11), # size
                        strides = (6,4), #step size
                        activation = 'relu',
                )) #output 51x62x 12
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0),(0,0)),
                        input_shape = input_shape))
                self.model.add(layers.Dropout(0.3))
                #output = 52x62x 12
                self.model.add(layers.MaxPool2D(pool_size=2))
                #output 26x31x 12
                self.model.add(layers.DepthwiseConv2D(
                3, # size
                strides = (1,1), #step size
                depth_multiplier = 2,
                activation = 'relu',
                )) #output 23x28x12
                self.model.add(layers.ZeroPadding2D(
                        padding = ((1,0),(0,0)),
                        input_shape = input_shape))
                #output 24x28x12
                self.model.add(layers.MaxPool2D(pool_size=2))
                #output 12x14x12
                self.model.add(layers.Flatten())
                #output: 4032
                self.model.add(layers.Dense(1024, activation = 'relu'))
                self.model.add(layers.Dense(256, activation = 'relu'))
                self.model.add(layers.Dense(64, activation = 'relu'))
                self.model.add(layers.Dense(14, activation = 'softmax'))
                self.loss = losses.CategoricalCrossentropy()
                self.optimizer = optimizers.SGD(learning_rate = 0.0001, momentum=0.5, decay = 0.0001/100)
                self.model.compile(
                        loss = self.loss,
                        optimizer = self.optimizer,
                        metrics = ["accuracy"]
                )
        def __str__(self):
                self.model.summary()
                return ""
for person in class_names:
    # Get the first image of that person and set it up
    img = cv2.imread(f'images/{person}/0{person}.png' )
    img = cv2.resize(img, (320, 258))
    img = utils.img_to_array(img)
    img = img[tf.newaxis, ...]

    # Did checkpoints every 2 epochs up to 40.
    for k in range(2, 80, 2):
        # Set up the architecture and load in the checkpoint weights
        net = Net((320, 258, 3))
        # print(net)
        checkpoint = Checkpoint(net.model)
        checkpoint.restore(f'checkpoints/checkpoints_{k:02d}').expect_partial()
        # Get the first conv layer, feed the image and set it up for viewing
        filters = net.model.layers[3](img)[0]
        shape = filters.shape
        filters = filters.numpy()
        # Put all filters in one big mosaic image with 2 rows, padded by 
        #   20px gray strips.  
        # Scaling up the filters by 3x to make them easier to see
        cols = shape[2] // 2
        mosaic = np.zeros(
            (6*shape[0] + 20, 3*cols*shape[1] + (cols - 1)*20)
        )  
        # Print the filter max and average to screen so we can see how much
        #   the classification uses this filter.
        print(f'{person:>12} Chkpt {k:02d} Maxes:', end = ' ')
        second_str = '                      Avgs: '
        # Shape[2] = number of filters
        for i in range(shape[2]):
            # Get just one filter
            filter = filters[0:shape[0],0:shape[1],i]
            # Calculate and print max and avg
            maxes = []
            avgs = []
            for j in range(shape[0]):
                maxes.append(max(filter[j]))
                avgs.append(sum(filter[j])/len(filter[j]))
            print(f'{max(maxes):8.4f}', end = ' ')
            second_str += f'{sum(avgs)/len(avgs):9.4f}'
            print(filter.shape)
            # Triple the filter size to make it easier to see
            # filter = cv2.resize(filter, (3*shape[0], 3*shape[1]))
            print(filter.shape)
            # Rescale so the grayscale is more useful
            filter = filter / max(maxes) * 2
            # Locate the filter in the mosaic and copy the values in
            offset = ((i % 2)*(shape[0] + 20), (i // 2)*(shape[1] + 20))
            # filter = np.swapaxes(filter, 1, 0)
            # print(filter)
            mosaic[
                offset[0]:offset[0] + shape[0], 
                offset[1]:offset[1] + shape[1]] = filter
            # np.swapaxes(filter, 0, 1)
        print()
        print(f'{second_str}')
        # Make the gray stripes that separate the filters
        # Vertical Stripes
        for i in range(1, cols):
            start_vert_stripe = 3*i*shape[1] + (i - 1)*20
            mosaic[
                0:mosaic.shape[0], 
                start_vert_stripe:start_vert_stripe + 20] = np.ones(
                    (mosaic.shape[0], 20)) * 0.5  
        # Horizontal Stripe
        mosaic[3*shape[0]:3*shape[0] + 20, 0:mosaic.shape[1]] = np.ones(
                (20, mosaic.shape[1])) * 0.5
        # Display the image
        cv2.imshow(f'{person} Checkpoint {k}', mosaic)
        if chr(cv2.waitKey(0)) == 'q':
            quit()
        cv2.destroyAllWindows()