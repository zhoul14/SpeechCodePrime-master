#Test.py  
import numpy as np
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
    data = np.fromfile(filename,dtype=myDtype)
    L = len(data.shape)
    return data

def getFWTMP(dirName, stateNum, frameNum, fdim = 45):
    data = np.ones((frameNum, fdim), dtype="float32")
    label = np.ones((frameNum,1), dtype = "int32")
    p = 0
    for i in range(0,stateNum):
        data0 =getBiFile2Py(dirName+"\\%.4d.tmp"%(i))
        data0=data0.reshape(data0.shape[0]/fdim,fdim)
        q = data0.shape[0]
        data[p:q+p] = data0
        label[p:p+q] = i
        p = q + p
    print data.shape
    data = data.reshape(data.shape[0]*fdim)
    return data,label

def getArray(i):
    return i

def getModel(jsonName, h5Name):
    model = model_from_json(open(jsonName).read())
    model.load_weights(h5Name)
    return model

def getPrepare(jsonName, h5Name,features,fnum,fdim=45):
    model = getModel(jsonName,h5Name)
    features = features.reshape(features.shape[0]/fdim,fdim)
    p = model.predict_proba(features,batch_size=100,verbose=1)
    return p

def makeModel(jsonName, h5Name, dirName, stateNum, frameNum, fdim=45):
    data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
    data = data.reshape(data.shape[0]/fdim, fdim)
    #label =  np_utils.to_categorical(label, stateNum)
    model = Sequential()
    model.add(Dense(300,input_dim=fdim,activation='tanh',W_regularizer=l2(0.01)))
    model.add(Dense(480,activation='tanh',W_regularizer=l2(0.00)))
    model.add(Dense(800,activation='tanh',W_regularizer=l2(0.01)))
    model.add(Dense(857,activation='softmax'))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
    print "momdel compling"
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.1)
    model.save_weights(h5Name)
    json_string = model.to_json()
    open(jsonName,'w').write(json_string)

def trainModel(jsonName, h5Name, dirName, stateNum, frameNum, fdim=45):
    data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
    data = data.reshape(data.shape[0]/fdim, fdim)
    model = getModel(jsonName, h5Name)
    model.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.1)
    model.save_weights(h5Name)
    json_string = model.to_json()
    open(jsonName,'w').write(json_string)



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