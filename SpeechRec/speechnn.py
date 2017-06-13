# -*- coding: utf-8 -*-

import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,model_from_json
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.regularizers import l2,
import random
from os.path import join, getsize

## 获取文件架的大小
def getdirsize(dir):
    size = 0L
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files if os.path.splitext(name)[1] =='.tmp'])
        
    return size
    
##清除.tmpx文件
def clearMultiTmpFiles(dir, n):
    for root, dirs, files in os.walk(dir):
        for name in files:
            for i in range(0,n):
                if os.path.splitext(name)[1] =='.tmp%d'%i:
                    os.remove(join(root,name))


def errorStr(functionName):
    return '[python error]:'+functionName

#如果训练语音特征矢量过大，不能一次训练完成。则使用这个函数进行打乱。平均分成n份。再进行一份一份的训练
#这就是打乱的程序，比如n=2.那么会把FWTMP中的每个tmp，例如3201.tmp，分成3201.tmp0,3021.tmp1。以此类推。之后每次训练的时候，会先训练tmp0的文件再训练tmp1的文件。
def makeFWTMPShuffle(dirName, n, stateNum,fdim = 45):
    print n,'clear old fwtmpshuffle file.'
    clearMultiTmpFiles(dirName,n)
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


##c++交互函数
def getCpp2Py(data):
    print data
    return data


def getTxtFile2Py(filename):
    PrimeData = np.loadtxt(filename)
    return PrimeData

##获取二进制的文件内容，因为.tmp都是2进制 保存
def getBiFile2Py(filename, fdim=45, myDtype="float32"):
    if os.path.exists(filename) == False:
        print filename,"does not exist"
        return []
    data = np.fromfile(filename,dtype=myDtype)
    return data

## 获取FWTMP的特征矢量，返回data和label。
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

## 获取tmpx为后缀文件的特征矢量，比如appendNum=1，那就是获取所有.tmp1的特征矢量，存到data和label中
def getFWTMPmulti(dirName, stateNum, frameNum, appendNum,fdim = 45):
    data = np.ones((frameNum, fdim), dtype="float32")
    label = np.ones((frameNum,1), dtype = "int32")
    p = 0
    print data.shape
    for it in range(0,int(stateNum)):
        data0 =getBiFile2Py(dirName+"\\%.4d.tmp%d"%(it,appendNum),fdim)
        if len(data0) == 0:
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

## 获取一个全局的gModel，这样可以不用每次getPrepare的时候再需要重新load model
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

## 进行概率预计算，用的是上个函数中获取的全局gModel
def getPrepare(features,fnum,fdim=45):
    features = features.reshape(features.shape[0]/fdim,fdim)
    p = gModel.predict_proba(features,batch_size=100,verbose=0)
    return p

## 给输出文件名和FWTMP文件名以及状态数，总的参与训练的帧数和特征维度。
## 训练出一个全新的Model
def makeModel(outJName, outHName,dirName, stateNum, frameNum, fdim=45):
    print "make model"
    multiStep = 0
    epoch = 3##循环三次，可以改
    print epoch
    totalFrameSize = getdirsize(dirName)## 获取总训练帧的内存大小，如果超过6000000000。可能无法一次完成识别。就需要分成n份。
    if  totalFrameSize>= 6000000000:
        multiStep = 1
        n = totalFrameSize/6000000000 + 1## n表示分的份数
        TotalFrameList = makeFWTMPShuffle(dirName, n, stateNum, fdim)##  将FWTMP里的.tmp分成n份，保存到FWTMP/XXX.tmp0,FWTMP/XXX.tmp1，FWTMP/XXX.tmp2.......FWTMP/XXX.tmpn
        print TotalFrameList
        print 'use multi data'
    else:
        data,label=getFWTMP(dirName, stateNum, frameNum, fdim)## 数据量可以让内存一次训练完成
        data = data.reshape(data.shape[0]/fdim, fdim)
    print "data completed"
    model = Sequential()##神经网络的构造开始
    model.add(Dense(600,input_dim = fdim,activation='tanh'))
    model.add(PReLU())
    model.add(Dense(2048))
    model.add(PReLU())
    model.add(Dense(2048))
    model.add(PReLU())
    model.add(Dense(4096))
    model.add(PReLU())
    model.add(Dense(3206,activation='softmax'))## 构造结束
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)## 优化方法SGD和学习率等
    print "momdel compling"
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])## 模型编译，设置目标函数等
    if multiStep != 1 :## 多次训练和单词训练的flag
        model.fit(data,label,batch_size=100, nb_epoch=epoch,shuffle=True,verbose=1)## 单词训练
    else :
        for iter in range(epoch):
            print 'train echo[%d]'%iter
            for i in range(0,n):
                data,label=getFWTMPmulti(dirName,stateNum, TotalFrameList[i],i,fdim)## 获取.tmpi的data和label
                data = data.reshape(data.shape[0]/fdim, fdim)
                model.fit(data,label,batch_size=100, nb_epoch=1,shuffle=True,verbose=1)## 训练.tmpi的data和label
                model.save_weights(outHName,overwrite=True)## 保存模型参数
                json_string = model.to_json()
                open(outJName,'w').write(json_string)    ## 保存模型结构
    print outHName,outJName
    model.save_weights(outHName,overwrite=True)
    json_string = model.to_json()
    open(outJName,'w').write(json_string)##最终保存一次
    getModel(outJName, outHName)

## 给新输出文件名，和老的模型文件，FWTMP文件名以及状态数，总的参与训练的帧数和特征维度。
## 过程除了model是装载的模型文件，而不是生成新的模型，其余代码和makemodel基本相同
def trainModel(jsonName, h5Name, outJName, outHName,dirName, stateNum, frameNum, learnRate, fdim=45):
    print "data loading!"
    print jsonName, h5Name, dirName, stateNum, frameNum, fdim
    multiStep = 0
    epoch = 1
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
    sgd = SGD(lr=learnRate , decay=1e-6, momentum=0.9, nesterov=True)
    time.sleep(10)
    print "json model:"
    Model = model_from_json(open(jsonName).read())##load 模型结构
    print "json model end"
    Model.load_weights(h5Name)##load 模型参数
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
    Model.save_weights(outHName,overwrite=True)
    json_string = Model.to_json()
    open(outJName,'w').write(json_string)
