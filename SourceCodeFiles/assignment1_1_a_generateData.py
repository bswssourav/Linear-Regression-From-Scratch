import numpy as np
import random as rd
import math as mt
import pandas as pd
import xlwt
import matplotlib.pyplot as plt
import xlrd
from xlwt import Workbook
from array import *

dataxl = xlwt.Workbook(encoding="utf-8")

sheet1 = dataxl.add_sheet('sheet 1')
mean=0
sd=0.3

w, h = 2, 10;
data = [[0 for x in range(w)] for y in range(h)]
sheet1.write(0, 0, 'x')
sheet1.write(0, 1, 'y')
for i in range(0, 10):
    x=rd.uniform(0, 1)
    yi=0
    #print("x= ",x)
    yi=np.sin((2 * mt.pi * x))
    #print("y before", y)

    ya=yi+np.random.normal(0,sd)
    #print("y after", ya)
    data[i][0]=x
    data[i][1]=ya
    sheet1.write(i+1, 0, x)
    sheet1.write(i+1, 1, ya)
    #print(data)

#print(data)
dataxl.save("C:\\Users\\Sourav Biswas\\Desktop\\sampleData.xls") #change the path

df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\sampleData.xls","sheet 1")  #change the path


print("the Generated datas ------\n",df)

