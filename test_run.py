from Model3 import *
import csv
import time


def test(filename, alpha, beta, gamma, threshold):
    start_time = time.time()
    data = []

    filename = 'sample.csv'
    csvfile = open(filename,encoding='utf-8')
    reader = csv.reader(csvfile)
    reader.__next__()  # Ignoring the header
    for row in reader:
        data.append([row[3], row[4], int(row[5])])

    #threshold = 0.5
    #alpha = 0.4
    #beta = 0.3
    #gamma = 0.3

    prediction = []
    for d in data:
        #print(d[0])
        start_time = time.time()
        result, score = predict(d[0], d[1], threshold, alpha, beta, gamma)
        prediction.append(result)
        #print('Actual : {0:2d}, Predict : {1:2d}, score : {2:6.2f}, run time : {3:6.4f} seconds'.format(d[2], result, score, time.time() - start_time))
    return prediction, data



def performance(prediction, data):
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
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP/(TP + FP)
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP/(TP + FN)
    if precision + recall == 0:
        return 0, 0, 0
    F = 2*precision*recall/(precision + recall)

    #print("threshod : {0:4.2f}, alpha : {1:4.2f}, beta : {6:4.2f}, beta : {7:4.2f}, Precision : {2:6.4f}, Recall : {3:6.4f}, F : {4:6.4f}, Run time : {5:6.4f} seconds".format(threshold, alpha, precision, recall, F, time.time()-start_time, beta, gamma))
    return precision, recall, F


#prediction, data = test('sample.csv', 0.4, 0.3, 0.3, 0.5)
#p, r, f = performance(prediction, data)

#print(p, r, f)
