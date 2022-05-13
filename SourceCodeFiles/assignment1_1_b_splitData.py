import numpy as np
import random as rd
import math as mt
import pandas as pd
import xlwt
import matplotlib.pyplot as plt
import xlrd
from xlwt import Workbook


df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\sampleData.xls","sheet 1") #change the path


#print(df)



#newData=rd.shuffle(df)
trainPer=int((len(df))*(70/100))
#print("trainper==",trainPer)
train_data = df[:trainPer]
test_data =df[trainPer:]
print("Train Data set == \n",train_data)
print("Test Data Set == \n",test_data)