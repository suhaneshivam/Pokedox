from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K

class SmallVggNet:
    @staticmethod
    def build(width ,height ,depth ,classes):
        model = Sequential()
        inputShape = (height ,width ,depth)
        chamDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (depth ,height ,width)
            chamDim = 1

        model.add(Conv2D(32 ,(3 ,3) ,padding="same" ,input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chamDim))
        model.add(MaxPooling2D(pool_size = (2 ,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64 ,(3 ,3) ,padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chamDim))
        model.add(Conv2D(64 ,(3 ,3) ,padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chamDim))
        model.add(MaxPooling2D(pool_size = (2 ,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
