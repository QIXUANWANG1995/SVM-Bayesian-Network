# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 09:12:41 2016

@author: SillyGuy
"""

import numpy
#import DealData as dData
#使用非连续的数据的一个分量计算出概率
def calPreProForOneComponent(dataLine, labels, labelTypes, dataMark):
    #dataMat = numpy.transpose(dataMat)
    
    #resultPro = list(list())
    #resultCount = list(list(list()))
    #decodeData, dataMark = dData.decodeLabelType(dataLine)
    resultPro = [[0]*len(labelTypes)]*len(dataMark)
    for index, data in enumerate(dataLine):
        resultPro[data][labels[index]] += 1;
    for i in xrange(len(dataMark)):
        for j in xrange(len(labelTypes)):
            resultPro[i][j] /= len(dataLine)
            
    '''
    for index, dataLine in enumerate(dataMat):
        resultLinePro = list(list())
        #计算出该分量的标签
        decodeData, labelMark = dData.decodeLabelType(dataLine)
        dataLableMarks.append(labelMark)#存放该分量的LabelMark
        for 
        #for i in xrange(0,len(labelMark)):
         #   counter = decodeData.count(i)
    '''
    return resultPro

#使用非连续的数据计算出概率
def calPreProForAll(dataSet, labels, labelTypes, dataSetMarks):
    dataSet = numpy.transpose(dataSet)
    resultPros = list(list(list()))
    for index, dataline in enumerate(dataSet):
        resultPro = calPreProForOneComponent(dataline, labels, labelTypes, dataSetMarks[index])
        resultPros.append(resultPro)
        
    return resultPros

#使用先验概率计算后验概率
def calPostPro(dataSetTest, prePros, dataMarks, labelTypes):
    postPros = list(list())
    for dataline in dataSetTest:
        #处理一个数据
        #print dataline
        postPro = numpy.array([0.0]*len(labelTypes))
        for index, prePro in enumerate(prePros):
            #处理一个数据的一个分量
            #print postPro
            #print prePro[dataline[index]]
            postPro = postPro + numpy.array(prePro[dataline[index]])
        postPros.append(postPro)
    return postPros
        

#计算label未知数据的label
def BayesCal(dataSetTest, prePros, dataMarks, labelTypes):
    resultLabels = []
    postPros = calPostPro(dataSetTest, prePros, dataMarks, labelTypes)
    for postPro in postPros:
        label = numpy.argmax(postPro)
        resultLabels.append(label)
    return resultLabels;
    
#将SVM结果和贝叶斯方法结合起来
def combineSVMandBayse(BayesPostPros, svmErrorPros, svmResultLabels):
    BayesPostPros = numpy.array(BayesPostPros)
    for index, postPros in enumerate(BayesPostPros):
        BayesPostPros[index] = postPros + numpy.array(svmErrorPros[int(svmResultLabels[index])])*1000
    
    return BayesPostPros
    
#使用后验概率获得结果
def calResult(postPros, labels):
    resultLabels = []
    counter = 0;
    for index, postPro in enumerate(postPros):
        label = numpy.argmax(postPro)
        resultLabels.append(label)
        if (label == labels[index]):
            counter += 1;
    #accuracy = counter / len(labels)
    return resultLabels