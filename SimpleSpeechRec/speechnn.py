#Test.py  
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.regularizers import l2, activity_l2
import random
def HelloWorld():  
    print "Hello World"

def add(a, b):  
    return a+b  

def TestDict(dict):  
    print dict  
    dict["Age"] = 17  
    return dict  

class Person:  
    def greet(self, greetStr):  
        print greetStr  

def getCpp2Py(data):
    print data
    return data

def getTxtFile2Py(filename):
    PrimeData = np.loadtxt(filename)
    return PrimeData

def getBiFile2Py(filename, fdim=45, myDtype="float32"):
    if os.path.exists(filename) == False:
        print filename,"does not exist"
        print "[error]!!!!!!!"
    data = np.fromfile(filename,dtype=myDtype)
    #print data.shape
    return data

def getFWTMP(dirName, stateNum, frameNum, fdim = 45):
    data = np.ones((frameNum, fdim), dtype="float32")
    label = np.ones((frameNum,1), dtype = "int32")
    p = 0
    print data.shape
    for it in range(0,int(stateNum)):
        data0 =getBiFile2Py(dirName+"\\%.4d.tmp"%(it),fdim)
        data0=data0.reshape(data0.shape[0]/fdim,fdim)
        q = data0.shape[0]
        data[p:q+p] = data0
        label[p:p+q] = it
        p = q + p
    
    data = data.reshape(data.shape[0]*fdim)
    return data,label

def getArray(i):
    return i

def getModel(jsonName, h5Name):
    print "fc"
    global gModel
    gModel = model_from_json(open(jsonName).read())
    gModel.load_weights(h5Name)
    gModel.summary()
    return gModel

def getPrepare(jsonName, h5Name,features,fnum,fdim=45):
    #model = getModel(jsonName,h5Name)
    #print jsonName, h5Name,features.shape,fnum,fdim
    features = features.reshape(features.shape[0]/fdim,fdim)
    #print gModel
    p = gModel.predict_proba(features,batch_size=100,verbose=0)
    #print p.shape
    return p

def makeModel(jsonName, h5Name, dirName, stateNum, frameNum, fdim=45):
    print "make model"
    #print dirName
    #print stateNum
    #print frameNum
    #print fdim
    data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
    data = data.reshape(data.shape[0]/fdim, fdim)
    #label =  np_utils.to_categorical(label, stateNum)
    print "data completed"
    model = Sequential()
    model.add(Dense(600,input_dim=fdim,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dense(1000,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dense(1200,activation='tanh',W_regularizer=l2(0.01)))
    model.add(Dense(800,activation='tanh',W_regularizer=l2(0.01)))
    model.add(Dense(857,activation='softmax'))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
    print "momdel compling"
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True)
    model.save_weights(h5Name,overwrite=True)
    json_string = model.to_json()
    open(jsonName,'w').write(json_string)
    getModel(jsonName, h5Name)
    #gModel = model


def trainModel(jsonName, h5Name, dirName, stateNum, frameNum, fdim=45):
    print "data loading!"
    print jsonName, h5Name, dirName, stateNum, frameNum, fdim
    data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
    print "train-------gModel"
    data = data.reshape(data.shape[0]/fdim, fdim)
    print "train-------gModel"
    #model = getModel(jsonName, h5Name)
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    gModel.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    gModel.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True)
    gModel.save_weights(h5Name,overwrite=True)
    json_string = gModel.to_json()
    open(jsonName,'w').write(json_string)
    #getModel(jsonName, h5Name)



# jsonName='model.json'
# h5Name='model.h5'
# dirName='FWTMP'
# stateNum=857
# frameNum=6025483
# fdim=45
#print add(5,7)  

# data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
# data = data.reshape(data.shape[0]/fdim, fdim)
#label =  np_utils.to_categorical(label, stateNum)
# model = Sequential()
# model.add(Dense(300,input_dim=45,activation='tanh',W_regularizer=l2(0.01)))
#model.add(Dense(300,activation='tanh',W_regularizer=l2(0.01)))
# model.add(Dense(480,activation='tanh',W_regularizer=l2(0.00)))
#model.add(Dense(800,activation='tanh',W_regularizer=l2(0.01)))
# model.add(Dense(857,activation='softmax'))
# sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
# print "momdel compling"
# model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
# model.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.1)
#a = raw_input("Enter To Continue...")
#getModel("model.json","model.h5")
#trainModel('model.json','model.h5','FWTMP2','857',189700,495)