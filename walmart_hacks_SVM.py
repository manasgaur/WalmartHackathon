import csv
import numpy as np
import pandas as pd
from itertools import izip
from sklearn import svm
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
WDitemid=[]
WDClabels=[]
shelves=[]
f1=open('shelves_predict.txt','a')
f2=open('itemsid.txt','a')
with open('/Users/manasgaur/Desktop/MyApp/Mutliprocessing/swalmart/SVM_dataset.csv','r') as f:
    freader=csv.reader(f,delimiter='\t')
    c=0
    for row in freader:

        if c==0:
            c=c+1
            continue;
        else:
            c=c+1
            WDitemid.append(int(row[0].split(',')[1]))
            WDClabels.append(int(row[0].split(',')[0]))
            shelves.append(int(row[0].split(',')[2]))
#labelling the shelves
uniq_tags=list(set(shelves))
labels={}
#f3=('open_shelves.txt','w')
for i in range(len(uniq_tags)):
    labels[uniq_tags[i]]=i

print labels
#print len(WDClabels)
numSVM=list(set(WDClabels))
testWDitemid=[]
testWDClabels=[]
with open('/Users/manasgaur/Desktop/MyApp/Mutliprocessing/swalmart/SVM_testset.csv','r') as f:
    freader=csv.reader(f,delimiter='\t')
    c=0
    for row in freader:

        if c==0:
            c=c+1
            continue;
        else:
            c=c+1
            testWDClabels.append(int(row[0].split(',')[0]))
            testWDitemid.append(int(row[0].split(',')[1]))
#print numSVM
results=[]
#f-score, precision, recall and accuracy metric
f_total=0
p_total=0
r_total=0
a_total=0
for i in numSVM:
    #creating a subset dataset
    subset_item=[]
    subset_labels=[]
    subset_shelves=[]
    test_subset_item=[]
    test_subset_labels=[]
    for j in range(len(WDClabels)):
        if WDClabels[j]==i:
            subset_item.append(WDitemid[j])
            subset_labels.append(i)
            subset_shelves.append(labels[shelves[j]])
    for k in range(len(testWDClabels)):
        if testWDClabels[k]==i:
            test_subset_item.append(testWDitemid[k])
            test_subset_labels.append(i)

    testset=pd.DataFrame({'test_Clabels': test_subset_labels,
                          'test_ID' : test_subset_item})
    #print testset
    dataset=pd.DataFrame({'Tags': subset_shelves,
                          'Clabels':subset_labels,
                          'ID':subset_item})
    #print dataset
    r_state=np.random.RandomState(0)
    X_train, X_test, y_train, y_test = train_test_split( zip(dataset['ID'],dataset['Clabels']), dataset['Tags'], test_size=0.1, random_state=r_state)
    #print X_train
    model = svm.SVC(kernel='rbf', probability=True,random_state=r_state)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    f_total+=(f1_score(y_test, y_pred, average="macro"))
    p_total+=(precision_score(y_test, y_pred, average="macro"))
    r_total+=(recall_score(y_test, y_pred, average="macro"))
    test_output=model.predict(testset)
    results.append(test_output)
    #print testset['test_ID']
    #print (pXtest==y_test)
    scores = cross_val_score(model, zip(dataset['ID'],dataset['Clabels']),dataset['Tags'])
    a_total+=scores.mean()+scores.std()
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
for x in results:
    for i in x:
        f1.write(str(i))
        f1.write("\n")
print 'f_score',(float(f_total)/5) # 5 is the number of clusters
print 'recall',(float(r_total)/5)
print 'precision',(float(p_total)/5)
print 'accuracy',(float(a_total)/5)
f1.close()
f2.close()
