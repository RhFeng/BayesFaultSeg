from faultSeg_classes import DataGenerator
#import matplotlib.pyplot as plt
import numpy as np


# training image dimensions
n1, n2, n3 = 128, 128, 128
params = {'batch_size': 1,
          'dim':(n1,n2,n3),
          'n_channels': 1,
          'shuffle': True}

tdpath = 'data/train/seis/'
tfpath = 'data/train/fault/'

vdpath = 'data/validation/seis/'
vfpath = 'data/validation/fault/'
tdata_IDs = range(200) #200
vdata_IDs = range(20)
training_generator   = DataGenerator(dpath=tdpath,fpath=tfpath,data_IDs=tdata_IDs,**params)
validation_generator = DataGenerator(dpath=vdpath,fpath=vfpath,data_IDs=vdata_IDs,**params)

#%%
from unet3_dropout import *
from keras import callbacks
#from unet3 import unet
from unet3_dropout import unet
import os

#K.set_image_data_format('channels_last')
model_name = 'fault'
model_dir     = os.path.join('check', model_name)
csv_fn        = os.path.join(model_dir, 'train_log.csv')
checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}.hdf5')

model = unet()
checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=False)
csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')
tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=2,
                                        write_graph=True, write_grads=True, write_images=True)
history = model.fit_generator(
                        generator=training_generator,
                        validation_data=validation_generator,
                        epochs=50,verbose=1,callbacks=[checkpointer, csv_logger, tensorboard])

print(history)
# serialize model to JSON
model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model3.h5")
print("Saved model to disk") 

#%%
## list all data in history
#print(history.history.keys())
#fig = plt.figure(figsize=(10,6))
#
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model accuracy',fontsize=20)
#plt.ylabel('Accuracy',fontsize=20)
#plt.xlabel('Epoch',fontsize=20)
#plt.legend(['train', 'test'], loc='center right',fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=18)
#plt.tick_params(axis='both', which='minor', labelsize=18)
#plt.show()
#
## summarize history for loss
#fig = plt.figure(figsize=(10,6))
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss',fontsize=20)
#plt.ylabel('Loss',fontsize=20)
#plt.xlabel('Epoch',fontsize=20)
#plt.legend(['train', 'test'], loc='center right',fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=18)
#plt.tick_params(axis='both', which='minor', labelsize=18)
#plt.show()

##%%
#from keras.models import load_model
#from unet3_dropout import *
##import tensorflow as tf
##from tf.keras.model import model_from_json
#
## load json and create model 
#json_file = open('model3.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("check/fault/checkpoint.02.hdf5")
#print("Loaded model from disk")
#
#
#gx,m1,m2,m3 = np.fromfile("data/validation/seis/6.dat",dtype=np.single),128,128,128
#gx = gx-np.min(gx)
#gx = gx/np.max(gx)
#gx = gx*255
#k = 50
#x = np.reshape(gx,(1,128,128,128,1))
#Y = loaded_model.predict(x,verbose=1)
#print(Y.shape)

# Y1 = Y[0]
# Y2 = Y[1]
# Y3 = Y[2]
# Y4 = Y[3]
# Y5 = Y[4]
#Y6 = Y[5]
##%%
#fig = plt.figure(figsize=(10,10))
#plt.subplot(1, 2, 1)
#imgplot1 = plt.imshow(np.transpose(x[0,k,:,:,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)
#plt.subplot(1, 2, 2)
#imgplot2 = plt.imshow(np.transpose(Y[0,k,:,:,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)



