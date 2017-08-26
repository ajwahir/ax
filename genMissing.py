#To generate missing values of the income
import csv
import numpy as np

salTrain = []

with open('modTrain.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
        	salTrain.append(row)

salTrain.remove(salTrain[0])
# get nonzero rows
nzSalTrain=[]
count=-1
for i in range(0,len(salTrain)):
	if float(salTrain[i][2])<1:
		continue
	else:
		count=count+1
		nzSalTrain.append(salTrain[i])
		nzSalTrain[count].append(float(salTrain[i][2]))
		nzSalTrain[count].remove(nzSalTrain[count][2])
