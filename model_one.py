import numpy as np
from motion import motion_profile
from plot import draw_plots
from model import Model
import pandas as pd

import keras.models
# import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
# import tensorflo




# def generate_model():
        
#         model = Sequential()
#         model.add(Dense(40, activation='relu'))
#         model.add(Dropout(0.25))
#         model.add(Dense(40, activation='relu'))
#         model.add(Dropout(0.25))
#         model.add(Dense(1, activation='relu'))
#         model.compile(optimizer='rmsprop',
#                 loss=tf.keras.losses.MeanSquaredError(),
#                 metrics=['mse'])
#         return model

# def compile(model, lr=0.001):
#         model.compile(optimizer='rmsprop',
#                 loss=tf.keras.losses.MeanSquaredError(),
#                 metrics=['mse'])

# def fit(self, model, XTrain, YTrain, MAX_EPOCHS):
#             # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#             #                                         patience=patience,
#             #                                         mode='min')
#             # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#             self.logger.info("Training is started ... ")
#             history = model.fit(XTrain, YTrain, epochs=MAX_EPOCHS,
#                     #  validation_data=window.val,
#                   #    callbacks=[early_stopping,
#                    #  callback
#                          #       ]
#                      )
#             return history

# def summary(model):
#         with open('model_summary.txt','w') as file:
#            with redirect_stdout(file):
#                 model.summary()
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("position")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def get_compiled_model(all_features):
    # model = Sequential()
    # model.add(Dense(40,activation='relu'))
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(1))

    # model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['mse'])
    x = tf.keras.layers.Dense(40, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "mean_squared_error", metrics=["mse"])

    return model

from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


# def encode_categorical_feature(feature, name, dataset, is_string):
#     lookup_class = StringLookup if is_string else IntegerLookup
#     # Create a lookup layer which will turn strings into integer indices
#     lookup = lookup_class(output_mode="binary")

#     # Prepare a Dataset that only yields our feature
#     feature_ds = dataset.map(lambda x, y: x[name])
#     feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

#     # Learn the set of possible string values and assign them a fixed integer index
#     lookup.adapt(feature_ds)

#     # Turn the string input into integer indices
#     encoded_feature = lookup(feature)
#     return encoded_feature

dt=1e-5
t = np.arange(0, 1.1, dt)
u=len(t)
g = 9.81
k = 0.213
step=5*1e3
Sstep= 4*step

x1 = []
x2 = []
curr2 = []
curr1 = []
Ftotal = []
x1 = np.zeros(int(u))
x2 = np.zeros(int(u))
curr1 = np.zeros(int(u))
curr2 = np.zeros(int(u))
Ftotal = np.zeros(int(u))
j = 0

x1, x2, Ftotal, curr1, curr2 = motion_profile(dt, u, x1, x2, Ftotal, curr1, curr2)

# print(np.shape(curr1))
# # draw_plots(t, x1, x2, Ftotal, curr1, curr2)
# curr1 = np.reshape(110000,1)
# curr2 = np.reshape(110000,1)
# x1 = np.reshape(110000,1)
# x2 = np.reshape(110000,1)
# Ftotal = np.reshape(110000,1)
# print(np.shape(curr1))

inp = [curr1, curr2, Ftotal, x2, x1]
# inp = np.reshape(110000,5)
print(np.shape(inp))
# for i in range(len(inp)):
#      for c1, c2, f, x2, x1 in zip(*inp):
#      # print(c1, c2, f, x2, x1)
#           inp[i] = [c1, c2, f, x2, x1]
nparray = np.array(inp)
transpose = nparray.transpose()
inp = transpose.tolist()
df = pd.DataFrame(inp, columns=('current1','current2','Ftotal','velocity','position'))
# df = df.transpose()
print(df)
print(df.shape)

val_data = df.sample(frac=0.2, random_state=1337)
train_data = df.drop(val_data.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_data), len(val_data))
)

train_ds = dataframe_to_dataset(train_data)
val_ds = dataframe_to_dataset(val_data)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

current1 = keras.Input(shape=(1,), name="current1")
current2 = keras.Input(shape=(1,), name="current2")
Ftotal = keras.Input(shape=(1,), name="Ftotal")
velocity = keras.Input(shape=(1,), name="velocity")

all_inputs = [
    current1,
    current2,
    Ftotal,
    velocity,
]

current1_encoded = encode_numerical_feature(current1, "current1", train_ds)
current2_encoded = encode_numerical_feature(current2, "current2", train_ds)
Ftotal_encoded = encode_numerical_feature(Ftotal, "Ftotal", train_ds)
velocity_encoded = encode_numerical_feature(velocity, "velocity", train_ds)

all_features = tf.keras.layers.concatenate(
    [
        current1_encoded,
        current2_encoded,
        Ftotal_encoded,
        velocity_encoded,
    ]
)


# x = tf.keras.layers.Dense(40, activation="relu")
# x = tf.keras.layers.Dropout(0.5)(x)
# output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(all_inputs, output)
# model.compile("adam", "mean_squared_error", metrics=["mse"])
model = get_compiled_model(all_features)

# target = df.pop('position')

# dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# for feat, targ  in dataset.take(5):
#      print ('features: {}, position: {}'.format(feat,targ))

# tf.constant(df['velocity'])

# train_dataset = dataset.shuffle(len(df)).batch(1)
# print(train_dataset)

# print(np.shape(inp))
# # print(np.shape(out))

# index = len(inp[1])/2
# # print(index)

# inpTrain = []
# inpTest = []
# outTrain = x1[1:int(index)]
# # outTest = x1[int(index)+1:]

# for i in range(4):
#      inpTrain.append(inp[i:int(index)])

# print(len(inpTrain[1]))
# print(np.shape(outTrain))

# Training 


MAX_EPOCHS = 4

# nnmodel = Model()
# model = nnmodel.generate_model()
# model.compile(model)
# model.fit(model, inpTrain, outTrain, MAX_EPOCHS)
# model.summary(model)


# model = Sequential()
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(4, activation='relu'))
# model.compile(optimizer='rmsprop',
#                 loss=tf.keras.losses.MeanSquaredError(),
#                 metrics=['mse'])

# history = model.fit(train_data, epochs=MAX_EPOCHS, validation_data=val_data)
# # model.evaluate(inpTest, outTest, verbose = 2)

# with open('model_summary.txt','w') as file:
#            with redirect_stdout(file):
#                 model.summary()

# model.save('model_1.h5')

# model = get_compiled_model()

keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
model.fit(train_ds, epochs=MAX_EPOCHS, validation_data=val_ds)
