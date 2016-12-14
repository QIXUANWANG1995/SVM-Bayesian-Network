# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 09:11:45 2016

@author: WuMingYou
"""

import numpy

import DealData as dData
import SVM
import BayesMethod as Bayes
############文件预处理部分############
print '开始文件预处理，将随机选取一定数量的数据作为测试，再在未选中数据中选择一定数量的数据作为数据集......'
#随机选取一组测试数据,并将没有选中的作为训练数据
(nochangeDataSet, conDataSet, symDadaSet, classType) = dData.readFile('data.csv')
print '总数据集个数为' , len(nochangeDataSet)
selData, disSelData = dData.selectDataRand(nochangeDataSet, 500)
print '测试集个数为', len(selData)
dData.saveFile(selData, 'TestData.csv')
selData, disSelData = dData.selectDataRand(nochangeDataSet, 5000)
print '训练数据集个数为', len(selData)
dData.saveFile(selData, 'TrainData.csv')
del nochangeDataSet, conDataSet, symDadaSet, classType, selData, disSelData


##############训练部分##############
print '开始训练...'
###一、读取文件，并进行数据预处理，为了方便起见，这里将测试数据的预处理一起做了###
print '开始读取文件...'
##读训练文件
(nochangeDataSet, conDataSet, symDadaSet, classType) = dData.readFile('TrainData.csv')
(nochangeDataSetTest, conDataSetTest, symDadaSetTest, classTypeTest) = dData.readFile('TestData.csv')
del nochangeDataSet, nochangeDataSetTest
print len(classType)
print len(classTypeTest)
print '读取文件结束'
###二、数据预处理###
print '开始数据预处理...'
##将数据label变为int类型（下标）,以方便处理
labels, labelMarks = dData.decodeLabelType(classType + classTypeTest)
labels = dData.changeClassType(labels)
labelsTest = labels[len(classType):]
labels = labels[:len(classType)]
#print svmLabelsTest
print '开始数据归一化...'
##将连续数据归一化
conDataSet = numpy.array(conDataSet+conDataSetTest, dtype=float)
conDataSet = dData.normrize(conDataSet)

print '开始PCA降维...'
##将连续数据使用PCA降维##
conDataSet = dData.pca(conDataSet, 10)
conDataSetTest = conDataSet[len(labels):]
conDataSet = conDataSet[:len(labels)]
conDataSet = numpy.mat(conDataSet, dtype=float)

print '将非连续的数据变为数字（下标）...'
##将非连续的数据变为数字（下标），以方便之后的处理
symDadaSet = symDadaSet + symDadaSetTest
symDadaSet = numpy.array(symDadaSet)
symDadaSet = numpy.transpose(symDadaSet);
symDadaSetDecoded = []
symDataSetMark = []
for symDataLine in symDadaSet:
    symDataLineDecoded, symDataLineMark = dData.decodeLabelType(symDataLine)
    symDadaSetDecoded.append(symDataLineDecoded)
    symDataSetMark.append(symDataLineMark)
symDadaSetDecoded = numpy.transpose(symDadaSetDecoded);
symDadaSetTestDecoded = symDadaSetDecoded[len(labels):]
symDadaSetDecoded = symDadaSetDecoded[:len(labels)]

##删除不再使用的数据
del classType, classTypeTest, symDadaSet, symDadaSetTest

print '对连续性数据使用SVM训练...'
###三、使用支持向量机对连续数据进行处理，找出判别平面###
#
C = 0.6
toler = 0.001
maxIter = 5
svmLabels = dData.castLabels(labels)
svmLabels = numpy.mat(svmLabels).T
svmClassifier = SVM.trainSVM(conDataSet, svmLabels, C, toler, maxIter, kernelOption = ('linear', 0))
print 'SVM训练结束'

###四、使用贝叶斯方法对离散数据进行处理，获得相应参数###
print '对无度量关系的数据使用贝叶斯方法...'
prePros = Bayes.calPreProForAll(symDadaSetDecoded, labels, labelMarks, symDataSetMark)
print '贝叶斯训练结束'
###五、将支持向量机和贝叶斯方法结合起来###
print '将支持向量机和贝叶斯方法结合起来...'
svmErrorPros = SVM.calSVMError(svmClassifier, conDataSet, svmLabels, labelMarks)
print '结合过程结束'

##############测试部分##############
print '开始进入测试阶段'
###一、文件读取###
print '文件已在之前读取，这儿不再读取'
#注：文件读取以及预处理已经在之前处理完成

###二、使用训练结果获得的参数进行决策分析
print '开始对测试数据进行判断'
svmLabelsTest = dData.castLabels(labelsTest)
svmLabelsTest = numpy.mat(svmLabelsTest).T
BayesPostPros = Bayes.calPostPro(symDadaSetTestDecoded, prePros, symDataSetMark, labelMarks)
accuracy, svmResultLabels = SVM.testSVM(svmClassifier, conDataSetTest, svmLabelsTest)
postPros = Bayes.combineSVMandBayse(BayesPostPros, svmErrorPros, svmResultLabels)
resultLabels = Bayes.calResult(postPros, labelsTest)

###三、显示决策结果
print '判断结束，准确率为：', 100.0*accuracy, '%'
