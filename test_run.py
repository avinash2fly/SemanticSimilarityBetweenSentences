from Model3 import *
import csv
import time

start_time = time.time()
data = []
i = 0

csvfile = open('sample.csv',encoding='utf-8')
reader = csv.reader(csvfile)
reader.__next__()  # Ignoring the header
for row in reader:
    data.append([row[3], row[4], int(row[5])])

threshold = 0.4
delta = 0.7
alpha = 0.9

prediction = []
for d in data:
    start_time = time.time()
    result, score = predict(d[0], d[1], threshold, delta, alpha)
    prediction.append(result)
    #print('Actual : {0:2d}, Predict : {1:2d}, score : {2:6.2f}, run time : {3:6.4f} seconds'.format(d[2], result, score, time.time() - start_time))



'''
   TP  |  FN
   ----------
   FP  |  TN
'''
TP = 0
FP = 0
FN = 0
for i in range(len(prediction)):
    if prediction[i] == 1:
        if data[i][2] == 1:
            TP += 1
        else:
            FP += 1
    elif prediction[i] == 0 and data[i][2] == 1:
        FN += 1

precision = TP/(TP + FP)
recall = TP/(TP + FN)
F = 2*precision*recall/(precision + recall)

print("threshod : {0:4.2f}, delta : {1:4.2f}, alpha : {6:4.2f}, Precision : {2:6.4f}, Recall : {3:6.4f}, F : {4:6.4f}, Run time : {5:6.4f} seconds".format(threshold, delta, precision, recall, F, time.time()-start_time, alpha))

