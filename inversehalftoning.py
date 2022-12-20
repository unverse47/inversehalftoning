# ============================================
############################################
#      Inverse halftoning
############################################


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle
# ----
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    GlobalAveragePooling2D, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model


def display(array1, array2):
    # Displays ten random images from each one of the supplied arrays.

    n = 6

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(32, 8))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def Get_psnr(array_list1, array_list2):
    # ArraySize = np.shape(array_list1)[1] * np.shape(array_list1)[2]
    rec = np.zeros(len(array_list1))
    for i in range(len(array_list1)):
        err = (array_list1[i] - array_list2[i]).flatten()
        sum(err ** 2) / len(err)
        rec[i] = -10 * np.log10(sum(err ** 2) / len(err))
    return rec


# ---- construct a Res_net and its supporting classes/functions

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size, strides=(1 if not downsample else 2),
               filters=filters, padding="same")(x)
    y = ReLU()(y)
    # y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, strides=1,
               filters=filters, padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1, strides=2, filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net():
    inputs = Input(shape=(128, 128, 1))  # binary halftone of size 128x128
    num_filters = 32

    t = Conv2D(kernel_size=7, strides=1, filters=32, padding="same")(inputs)
    t = ReLU()(t)

    t = Conv2D(kernel_size=5, strides=1, filters=32, padding="same")(t)
    t = ReLU()(t)

    num_blocks_list = [1, 1, 1, 1, 1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=0, filters=32)

    outputs = Conv2D(kernel_size=3, strides=1, filters=1, padding="same", activation="sigmoid")(t)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# =======================================
# main pgm starts here
# =======================================

#--- prepare data
datafolder = r'D:\study\dissertation\code\code'
with open(os.path.join(datafolder,'Dataset_for_IH'), 'rb') as filehandle:
    tmp = pickle.load(filehandle) # read the data as binary data stream
[(xtrain, ytrain),(xtest,ytest)] = tmp
del tmp


# display(xtrain, ytrain)  # display some selected train data & their target outputs

#--- Build the model
modelX = create_res_net() # or create_plain_net()
modelX.summary()


checkpoint_save_path = "./checkpoint/ResNet.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    modelX.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
#--- train the model for making o/p = i/p


loss_rec = list([])


history = modelX.fit(
    x=xtrain,     # we expect when i/p is x, the o/p is the same x
    y=ytrain,
    epochs=20,          # 50, <-- ori
    batch_size=8,
    shuffle=True,
    validation_data=(xtest, ytest),
    callbacks=[cp_callback]
)



# Show acc和loss curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

loss_rec = loss_rec + history.history['loss']

fig_res, ax_res = plt.subplots()  # show how loss drops & how accuracy improves
ax_res.plot(loss_rec)
ax_res.set_xlabel('epoch')
ax_res.set_ylabel('loss')
plt.show()


#--- produce inv halftoning result of training data & test data
predictions = modelX.predict(xtrain) # Get inv halftoning result with the current model
display(xtrain, predictions)
psnr_train = Get_psnr(ytrain, predictions)
print('average PSNR of training set : ',np.average(psnr_train), ' dB')

predictions = modelX.predict(xtest) # Get inv halftoning result with the current model
display(xtest, predictions)
psnr_test = Get_psnr(ytest, predictions)
print('average PSNR of testing set : ',np.average(psnr_test), ' dB')



'''
#--- save the current model weights of future use (when necessary)
datafolder = r'D:\study\dissertation\code\code'
with open(os.path.join(datafolder, 'InvH-model-ResNet-weight.data'), 'wb') as filehandle:    #$$
    pickle.dump(modelX.weights, filehandle) # store the data as binary data stream
with open(os.path.join(datafolder, 'InvH-model-ResNet-train_loss_rec.data'), 'wb') as filehandle:    #$$
    pickle.dump(loss_rec, filehandle) # store the data as binary data stream

# with open(os.path.join(datafolder,'model0-weight.data'), 'rb') as filehandle:
#     myList = pickle.load(filehandle) # read the data as binary data stream

#============================================

'''

print(xtrain.shape)
plt.imshow(xtest[3,:])
plt.gray()
plt.show()
prediction = modelX.predict(xtest)
plt.imshow(prediction[3,:])
plt.gray()
plt.show()

'''
img1 = xtest[3,:]
plt.imshow(img1)
print(img1.shape)
#img1 = img1.reshape(128,128)
#print(img1.shape)
plt.gray()
plt.show()
#img1 = tf.expand_dims(img1, 0)
prediction1 = modelX.predict(img1)
plt.imshow(prediction1)
plt.gray()
plt.show()

'''

image = plt.imread('DBShalftone128.tif')
image = tf.convert_to_tensor(image)
print(image.shape)
# image = tf.expand_dims(image, 0)
image = tf.expand_dims(image, -1)
print(image.shape)
plt.imshow(image)
plt.gray()
plt.show()
predictions = modelX.predict(image)
plt.imshow(predictions.reshape(128,128))
plt.gray()
plt.show()
imagetest = plt.imread('peppers128.tif')
image = np.array(imagetest)
# imagetest = tf.expand_dims(image, 0)
# imagetest = tf.expand_dims(image, -1)
plt.imshow(image)
plt.gray()
plt.show()
psnr = Get_psnr(imagetest, predictions)
print('average PSNR of peppers : ',np.average(psnr), ' dB')


