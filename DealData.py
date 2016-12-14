# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 09:11:54 2016

@author: SillyGuy
"""
import csv
import numpy
import random

def readFile(fileName='data.csv'):
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
    conDataSet = list(list())
    symDadaSet = list(list())
    #totalDataSet = list(list())
    nochangeDataSet = list(list())
    classType = list()
    dataTypeList = [1,0,0,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,
                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    for counter, line in enumerate(reader):  
        conData = list()
        symData = list()
        i = 0
        for data in line:
            #conData.append(data)
            #symData.append(data)
            if (i == 41):
                classType.append(data)
            elif (dataTypeList[i]):
                conData.append(data)
            else:
                symData.append(data)
            i += 1;
        nochangeDataSet.append(line)    
        conDataSet.append(conData)
        symDadaSet.append(symData)
        #totalDataSet.append(conData+symData)
        
        #if (counter >= 99):
        #    break
        #print conData
        #break;
        #print dataset[2]
    csvfile.close() 
    #print totalDataSet[0][0]
    return (nochangeDataSet, conDataSet, symDadaSet, classType)
    
    
#将数据归一化
def normrize(dataMat):
    #减去均值
    meanValue = numpy.mean(dataMat, axis=0)
    dataMat = dataMat - meanValue
    
    #除以标准差
    dataMat = numpy.transpose(dataMat)
    for dataLine in dataMat:
        #print dataLine
        standardVar = numpy.var(dataLine)
        standardVar = numpy.sqrt(standardVar)
        #print standardVar
        if (numpy.abs(standardVar - 0.000000) > 0.000001):
            for i, data in enumerate(dataLine):
                dataLine[i] = data/standardVar
    dataMat = numpy.transpose(dataMat)
    return dataMat
    
#PCA降维
def pca(dataMat, topNfeat=99):
    # 先对数据进行预处理，假设数据已经归一化了
    #meanValue = numpy.mean(dataMat, axis=0)
    #meanRemoved = dataMat - meanValue
    #print meanValue
    
    # 计算协方差矩阵
    covMat = numpy.cov(dataMat, rowvar=0)
    # 计算特征值和特征向量
    eigVals,eigVects = numpy.linalg.eig(numpy.mat(covMat))
    # 对特征值进行排序
    eigSort = numpy.argsort(eigVals)
    eigValid = eigSort[:-(topNfeat+1):-1]
    redEigVecs = eigVects[:,eigValid]
    #print eigVals
    #print eigSort
    #print eigValid
    #print redEigVecs
    #print redEigVecs.T
    # 计算最后转换后的矩阵
    lowDataMat = dataMat * redEigVecs
    return lowDataMat
    #reconMat = (lowDataMat * redEigVecs.T) + meanValue
    #return lowDataMat, reconMat, redEigVecs, meanValue
    
#
def saveFile(dataMat, fileName='data.csv'):
    csvfile = file(fileName, 'wb')
    writer = csv.writer(csvfile)
    for dataline in dataMat:
        writer.writerow(dataline)

    csvfile.close()
    #writer.writerows(dataMat)

    
#对没有度量关系的进行解析，变为下标型的
def decodeLabelType(labelList):
    labelDecodeList = []
    labelMarks = []
    for label in labelList:
        indicator = 0
        for i, labelMark in enumerate(labelMarks):
            if (label == labelMark):
                indicator = 1
                break;
        if (indicator):
            labelDecodeList.append(i)
        else:
            labelDecodeList.append(len(labelMarks))
            labelMarks.append(label)
    return labelDecodeList, labelMarks
  
#随机选取一组数据
def selectDataRand(dataSet, dataCount=100):
    selDataIndex = set()
    disSelData = list(list())
    selData = list(list())
    for counter in xrange(0, dataCount, 1):
        while (1):
            randNum = random.randint(0, len(dataSet)-1)
            if (randNum not in selDataIndex):
                selData.append(dataSet[randNum])
                selDataIndex.add(randNum)
                break
            
    for i, dataLine in enumerate(dataSet):
        if (i not in selDataIndex):
            disSelData.append(dataLine)
    return selData, disSelData
    
#将数据变为两两分问题
def changeClassType(labels):
    for i, label in enumerate(labels):
        if ((label-0.0) >= 0.00001):
            labels[i] = 1
    return labels
    
def castLabels(labels):
    newLabel = []
    for label in labels:
        if (label == 0):
            newLabel.append(-1.0)
        else:
            newLabel.append(1.0)
    return newLabel
    
def decastLabes(labels):
    newLabel = []
    for label in labels:
        if ((label+1.0)<0.00001):
            newLabel.append(0)
        else:
            newLabel.append(1)
    return newLabel
