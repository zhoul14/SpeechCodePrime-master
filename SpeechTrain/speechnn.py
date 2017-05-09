#Test.py  
import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.regularizers import l2, activity_l2
import random
from os.path import join, getsize

def getdirsize(dir):
    size = 0L
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files if os.path.splitext(name)[1] =='.tmp'])
        
    return size

def errorStr(functionName):
    return '[python error]:'+functionName

def makeFWTMPShuffle(dirName, n, stateNum,fdim = 45):
    print n
    FrameNumList = [0]*n
    for it in range(0,int(stateNum)):
        filename = dirName+"\\%.4d.tmp"%(it)
        if os.path.exists(filename) == False:
            print filename + 'does not exist!'
            continue
        data0 =getBiFile2Py(filename,fdim)
        data0 = data0.reshape(data0.shape[0]/fdim, fdim)
        interval = data0.shape[0]/n
        if interval == 0:
            interval = 1
        for j in range(0,n):
            if j == n -1:
                dataj = data0[interval*(n-1)::,:]
            else:
                dataj = data0[interval*j:interval*(j+1),:]
            FrameNumList[j]+=(dataj.shape[0])
            dataj2 = dataj.reshape(dataj.shape[0]*dataj.shape[1],1)
            if dataj2.shape[0] == 0:
                print "State:[",it,"]:",interval,'j:',j
            dataj2.tofile(dirName+"\\%.4d.tmp%d"%(it,j))
    f = open('log\\'+'FrameNumList.txt','w')
    f.write(str(FrameNumList))
    f.write('\n')
    f.close()
    return FrameNumList



def getCpp2Py(data):
    print data
    return data

def getTxtFile2Py(filename):
    PrimeData = np.loadtxt(filename)
    return PrimeData

def getBiFile2Py(filename, fdim=45, myDtype="float32"):
    if os.path.exists(filename) == False:
        print filename,"does not exist"
        print "[error]!!!!!!!Just Return"
        return []
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
        if len(data0) == 0:
            continue
        data0=data0.reshape(data0.shape[0]/fdim,fdim)
        q = data0.shape[0]
        data[p:q+p] = data0
        label[p:p+q] = it
        p = q + p
    
    data = data.reshape(data.shape[0]*fdim)
    return data,label

def getFWTMPmulti(dirName, stateNum, frameNum, appendNum,fdim = 45):
    data = np.ones((frameNum, fdim), dtype="float32")
    label = np.ones((frameNum,1), dtype = "int32")
    p = 0
    print data.shape
    for it in range(0,int(stateNum)):
        ##print dirName+"\\%.4d.tmp%d"%(it,appendNum)
        data0 =getBiFile2Py(dirName+"\\%.4d.tmp%d"%(it,appendNum),fdim)
        if len(data0) == 0:
            print 'fc'
            continue
        if (data0.shape[0]/fdim) * fdim != data0.shape[0]:
            print data0.shape[0]/fdim,fdim,data0.shape[0]
            print filename + "data0.shape mod fdim != 0!"
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
    try:
        print gModel
    except:
        global gModel
        print 'new gModel'
        pass
    print jsonName,h5Name
    gModel = model_from_json(open(jsonName).read())
    gModel.load_weights(h5Name)
    gModel.summary()

def getPrepare(jsonName, h5Name,features,fnum,fdim=45):
    #model = getModel(jsonName,h5Name)
    #print jsonName, h5Name,features.shape,fnum,fdim
    features = features.reshape(features.shape[0]/fdim,fdim)
    #print gModel
    p = gModel.predict_proba(features,batch_size=100,verbose=0)
    #print p.shape
    return p

def makeModel(jsonName, h5Name, outJName, outHName,dirName, stateNum, frameNum, fdim=45):
    print "make model"
    #print dirName
    #print stateNum
    #print frameNum
    #print fdim
    multiStep = 0
    epoch = 3
    print epoch
    totalFrameSize = getdirsize(dirName)
    if  totalFrameSize>= 6000000000:
        multiStep = 1
        n = totalFrameSize/6000000000 + 1
        TotalFrameList = makeFWTMPShuffle(dirName, n, stateNum, fdim)
        print TotalFrameList
        print 'use multi data'
    else:
        data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
        data = data.reshape(data.shape[0]/fdim, fdim)
    #label =  np_utils.to_categorical(label, stateNum)
    print "data completed"
    model = Sequential()
    model.add(Dense(600,input_dim = fdim,activation='tanh'))
    model.add(PReLU())
    #model.add(Dense(120,input_dim = fdim,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dropout(0.2))
    model.add(Dense(2048))
    model.add(PReLU())
    model.add(Dense(2048))
    model.add(PReLU())
    #model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(PReLU())
    #model.add(Dropout(0.5))
    #model.add(Dense(800))
    #model.add(PReLU())
    #model.add(Dropout(0.5))
    model.add(Dense(3206,activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    print "momdel compling"
    #model.compile(loss='sparse_mean_squared_error', optimizer=sgd,metrics=['accuracy'])    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    if multiStep != 1 :
        model.fit(data,label,batch_size=100, nb_epoch=epoch,shuffle=True,verbose=1)
    else :
        for iter in range(epoch):
            print 'train echo[%d]'%iter
            for i in range(0,n):
                data,label=getFWTMPmulti(dirName,stateNum, TotalFrameList[i],i,fdim)
                data = data.reshape(data.shape[0]/fdim, fdim)
                model.fit(data,label,batch_size=100, nb_epoch=1,shuffle=True,verbose=1)
                model.save_weights(outHName,overwrite=True)
                json_string = model.to_json()
                open(outJName,'w').write(json_string)    
    print outHName,outJName
    model.save_weights(outHName,overwrite=True)
    json_string = model.to_json()
    open(outJName,'w').write(json_string)
    getModel(outJName, outHName)
    #gModel = model


def makeLSTMModel(jsonName, h5Name, dirName, stateNum, frameNum, fdim=45):
    print "make model"
    #print dirName
    #print stateNum
    #print frameNum
    #print fdim
    data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
    data = data.reshape(data.shape[0]/fdim, 1,fdim)
    #label =  np_utils.to_categorical(label, stateNum)
    print "data completed"
    model = Sequential()
    model.add(LSTM(100,input_dim=45,input_length=fdim/45,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dense(1000,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dropout(0.5))
    #model.add(Dense(1200,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dropout(0.5))
    #model.add(Dense(800,activation='tanh',W_regularizer=l2(0.01)))
    #model.add(Dropout(0.5))
    model.add(Dense(857,activation='softmax'))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
    print "momdel compling"
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.fit(data,label,batch_size=100, nb_epoch=1,shuffle=True,verbose=1,show_accuracy=True)
    model.save_weights(h5Name,overwrite=True)
    json_string = model.to_json()
    open(jsonName,'w').write(json_string)
    getModel(jsonName, h5Name)
    #gModel = model

def trainModel(jsonName, h5Name, outJName, outHName,dirName, stateNum, frameNum, learnRate, fdim=45):
    print "data loading!"
    print jsonName, h5Name, dirName, stateNum, frameNum, fdim
    multiStep = 0
    epoch = 3
    totalFrameSize = getdirsize(dirName)
    print "train---model:prepare data"
    if  totalFrameSize>= 6000000000:
        multiStep = 1
        n = totalFrameSize/6000000000 + 1
        TotalFrameList = makeFWTMPShuffle(dirName, n, stateNum, fdim)
        print TotalFrameList
        print 'use multi data'
    else:
        data,label=getFWTMP(dirName, stateNum, frameNum, fdim)
        data = data.reshape(data.shape[0]/fdim, fdim)
    print "train-------gModel"
    #model = getModel(jsonName, h5Name)
    sgd = SGD(lr=learnRate , decay=1e-6, momentum=0.9, nesterov=True)
    time.sleep(10)
    print "json model:"
    Model = model_from_json(open(jsonName).read())
    print "json model end"
    Model.load_weights(h5Name)
    time.sleep(10)
    print Model
    print 'compute!!!!'
    try:
        Model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    except:
        print errorStr('compile')
    if multiStep != 1 :
        Model.fit(data,label,batch_size=100, nb_epoch=epoch,shuffle=True,verbose=1)
    else :
        for iter in range(epoch):
            print 'train echo[%d]'%iter
            for i in range(0,n):
                data,label=getFWTMPmulti(dirName,stateNum, TotalFrameList[i],i,fdim)
                print data.shape, label.shape
                data = data.reshape(data.shape[0]/fdim, fdim)
                Model.fit(data,label,batch_size=100, nb_epoch=1,shuffle=True,verbose=1,show_accuracy=True)        
                Model.save_weights(outHName,overwrite=True)
                json_string = Model.to_json()
                open(outJName,'w').write(json_string)
    #gModel.fit(data,label,batch_size=100, nb_epoch=3,shuffle=True,verbose=1,show_accuracy=True)
    Model.save_weights(outHName,overwrite=True)
    json_string = Model.to_json()
    open(outJName,'w').write(json_string)

#model.jsonfull_4mix_0.cb model.h5full_4mix_0.cb FWTMP 3206 11021275 495
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
# print "momde>>> 
#trainModel('model.jsonfull_8mix_0.cb','model.h5full_8mix_0.cb' ,'FWTMP' ,320#data loading!0.01,495)l compling"
# model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
# model.fit(data,label,batch_size=100, nb_epoch=2,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.1)
#a = raw_input("Enter To Continue...")
#getModel("model.json","model.h5")
#makeModel('model.json','model.h5','FWTMP2','857',189700,495)
# makeModel('model.json','model.h5','FWTMP2',857,2370263,45)
#makeModel('model.json','model.h5','FWTMP11',857,25912645/11,45*11)
#model.jsonfull_8mix_0.cb model.h5full_8mix_0.cb FWTMP 3206 32544675 495
#trainModel('model.jsonfull_8mix_0.cb','model.h5full_8mix_0.cb' ,'FWTMP' ,3206,32544675,0.01,495)