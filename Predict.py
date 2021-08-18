#%%
from keras.models import load_model
from unet3_dropout import *
import numpy as np
import matplotlib.pyplot as plt


#import tensorflow as tf
#from tf.keras.model import model_from_json

# load json and create model 
json_file = open('Dropout_000/model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Dropout_000/checkpoint.30.hdf5")
print("Loaded model from disk")

#%%

# set gaussian weights in the overlap bounaries
def getMask(os):
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)
    for k1 in range(os):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k1]
                sc[n1-k1-1][k2][k3]=sp[k1]
    for k1 in range(n1):
        for k2 in range(os):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k2]
                sc[k1][n3-k2-1][k3]=sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(os):
                sc[k1][k2][k3]=sp[k3]
                sc[k1][k2][n3-k3-1]=sp[k3]
    return sc

def predict(model, image, T):
    
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(T):
        p_hat.append(model.predict(image,verbose=0)[0])
    p_hat = np.array(p_hat)
    
    # mean prediction
    prediction = np.mean(p_hat, axis=0)
    # threshold mean prediction
    #prediction = np.where(prediction > 0.5, 1, 0)
    
    # estimate uncertainties
    aleatoric = np.mean(p_hat*(1-p_hat), axis=0)
    epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2

    return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic)


#a 3d array of gx[m1][m2][m3], please make sure the dimensions are correct!!!
#we strongly suggest to gain the seismic image before input it to the faultSeg!!!
gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),512,384,128
n1, n2, n3 = 128, 128, 128

list_stochastic_feed_forwards = [5,10,20,30,40,50]

os = 12 #overlap width
c1 = np.round((m1+os)/(n1-os)+0.5)
c2 = np.round((m2+os)/(n2-os)+0.5)
c3 = np.round((m3+os)/(n3-os)+0.5)
c1 = int(c1)
c2 = int(c2)
c3 = int(c3)
p1 = (n1-os)*c1+os
p2 = (n2-os)*c2+os
p3 = (n3-os)*c3+os
gx = np.reshape(gx,(m1,m2,m3))


result_dict = {}
for ind, num_stochastic_T in enumerate(list_stochastic_feed_forwards):
    print(ind)
    alea_list = []
    epis_list = []
    
    gp = np.zeros((p1,p2,p3),dtype=np.single)
    gy = np.zeros((p1,p2,p3),dtype=np.single)
    gy_ale = np.zeros((p1,p2,p3),dtype=np.single)
    gy_epi = np.zeros((p1,p2,p3),dtype=np.single)
    mk = np.zeros((p1,p2,p3),dtype=np.single)
    gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
    gp[0:m1,0:m2,0:m3]=gx
    sc = getMask(os)

    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                b1 = k1*n1-k1*os
                e1 = b1+n1
                b2 = k2*n2-k2*os
                e2 = b2+n2
                b3 = k3*n3-k3*os
                e3 = b3+n3
                gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
                gs = gs-np.min(gs)
                gs = gs/np.max(gs)
                gs = gs*255
                
                prediction, aleatoric, epistemic = predict(loaded_model, gs, T=num_stochastic_T)
                alea_list.append(np.mean(aleatoric))
                epis_list.append(np.mean(epistemic))

                
#                Y = loaded_model.predict(gs,verbose=1)
                Y = np.array(prediction)
#                gy[b1:e1,b2:e2,b3:e3]= Y[:,:,:]
                gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[:,:,:]*sc
                mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
                
                
                gy_ale[b1:e1,b2:e2,b3:e3]= gy_ale[b1:e1,b2:e2,b3:e3] + aleatoric[:,:,:]*sc
                
                gy_epi[b1:e1,b2:e2,b3:e3]= gy_epi[b1:e1,b2:e2,b3:e3] + epistemic[:,:,:]*sc
                
    result_dict.update({ '{}'.format(str(num_stochastic_T)) : 
    [num_stochastic_T, 1,
    np.mean(alea_list), np.std(alea_list),
    np.mean(epis_list), np.std(epis_list)]} )     
     
    gy = gy/mk
    gy_ale = gy_ale/mk
    gy_epi = gy_epi/mk

gy = gy[0:m1,0:m2,0:m3]
gy_ale = gy_ale[0:m1,0:m2,0:m3]
gy_epi = gy_epi[0:m1,0:m2,0:m3]

gy.tofile("data/prediction/f3d/"+"fp.dat",format="%4")

gy_ale.tofile("data/prediction/f3d/"+"fp_ale.dat",format="%4")

gy_epi.tofile("data/prediction/f3d/"+"fp_epi.dat",format="%4")

np.save("data/prediction/f3d/dict.npy",result_dict)

