# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 09:54:21 2016

@author: SillyGuy
"""

import DealData
import numpy

#测试文件写入
'''
(nochangeDataSet, totalDataSet, conDataSet, symDadaSet, classType) = DealData.readFile()
conDataSet = numpy.array(conDataSet, dtype=float)
DealData.saveFile(nochangeDataSet, 'newTemp.csv')
print 'final'
'''

#测试随机数据选取
'''
(nochangeDataSet, totalDataSet, conDataSet, symDadaSet, classType) = DealData.readFile()
selData = DealData.selectDataRand(conDataSet, 5)
print selData
print len(selData)
'''

#随机选取一组测试数据,并将没有选中的作为训练数据
(nochangeDataSet, totalDataSet, conDataSet, symDadaSet, classType) = DealData.readFile()
print '总数据集个数为' , len(nochangeDataSet)
selData, disSelData = DealData.selectDataRand(nochangeDataSet, 1000)
DealData.saveFile(selData, 'TestData.csv')
print '测试集个数为', len(selData)
DealData.saveFile(disSelData, 'TrainData.csv')
print '训练数据集个数为', len(disSelData)
del nochangeDataSet, totalDataSet, conDataSet, symDadaSet, classType
