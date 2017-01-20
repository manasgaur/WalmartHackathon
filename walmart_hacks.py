import csv
import re
import random
from  nltk.corpus import stopwords
from scipy.spatial.distance import cdist,pdist
#from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot as plt
count=0
Header=[]
data=[]
with open("/Users/manasgaur/Downloads/pstd/train.tsv",'r') as f:  #change the address local to the address of the data on your machine
    freader=csv.reader(f,delimiter="\t")
    for row in freader:
        if count==0:
            Header.append(row)
            count+=1
        else:
            data.append(row)
print Header
c=0
itemclassid=[]
itemid=[]
productname=[]
shelves=[]
for items in data:
    if items[Header[0].index('Product Name')] == "":# "" #and items[Header[0].index('Product Name')] == "":
        #print c
        c=c+1
    else:
        productname.append(items[Header[0].index('Product Name')])
        itemclassid.append(items[Header[0].index('Item Class ID')])
        itemid.append(items[Header[0].index('item_id')])
        shelves.append(items[Header[0].index('tag')])
        #print"\n"
        c=c+1
    if c==len(data):
        #print c
        break
newshelves=[]

print len(productname)
reduced_pname=[]
for items in productname:
    newitems=str(items).decode('utf-8').replace('.','').replace('!','')
    terms=re.split(':|-|' ' ',newitems)
    tempterms=[]
    for term in terms:
        if term not in stopwords.words('english'):
            tempterms.append(term.encode('utf-8'))
    reduced_pname.append(tempterms[0])
#print reduced_pname[0]
name_labels={}
uniq_name=list(set(reduced_pname))
for i in range(len(uniq_name)):
    name_labels[uniq_name[i]]=i
uniq_IC=set()
tofloat_item=[]
labels={}
templist=[] #dummy storage variable
for i in itemclassid:
    uniq_IC.add(i)
templist=list(uniq_IC)
for i in range(len(uniq_IC)):
    if uniq_IC:
            labels[uniq_IC.pop()]=i
#for k,v in labels.items():
 #   print k, v

label_ICid=[]
pname=[]
for i in range(len(itemid)):
    label_ICid.append(labels[itemclassid[i]])
    pname.append(name_labels[reduced_pname[i]])

#performing validation
dataset=pd.DataFrame({'IC': label_ICid,
                       'ID': pname})
######## This is section is to generate a small test set to validate clustering process ############
#X=random.sample(range(len(itemid)),1000)
#X=list(X)
#test_labelICid=[label_ICid[X[l]] for l in range(len(X))]
#test_itemid=[itemid[X[l]] for l in range(len(X))]
#testset=pd.DataFrame({'IC': test_labelICid,
 #                      'ID': test_itemid})
print len(templist)

#finding the appropriate K for the problem
K = range(1,50)
KM=[KMeans(n_clusters=k).fit(dataset) for k in K]
centroids=[k.cluster_centers_ for k in KM]

D=[cdist(dataset,C,'euclidean') for C in centroids]
cX=[np.argmin(d,axis=1) for d in D]
dist=[np.min(d,axis=1) for d in D]
Avg=[sum(d)/len(dataset) for d in dist]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(K,Avg,'r*-')
plt.show()

n_clusters=2
km=KMeans(n_clusters=n_clusters,max_iter=50000,n_init=100)
train_class_labels=km.fit_predict(dataset)
# finding the accuracy of the model
#diff=0
#for i in range(len(X)):
 #   diff+=abs(train_class_labels[X[i]]-test_class_labels[i])
#print float(diff)/len(test_class_labels)

arr=np.zeros(n_clusters)
for i in range(len(train_class_labels)):
    arr[train_class_labels[i]]+=1

plt.plot(np.arange(0, n_clusters),arr)
plt.show()

#we need to develop SVM for each cluster that is 7 SVM for multi-class labelling
#Intuition is that within the cluster the shelves will not be largely varying
WDitemid=[]
WDClabels=[]
for s,i,t in zip(shelves,pname,train_class_labels):
    if len(s)==1:
        s=s.replace('[','').replace(']','')
        newshelves.append(int(s))
        WDitemid.append(i)
        WDClabels.append(t)
    if len(s) > 1:
        s=s.replace('[','').replace(']','')
        terms=s.split(',')
        for term in terms:
            newshelves.append(int(term))
            WDitemid.append(i)  # here we are duplicating the tuples
            WDClabels.append(t)


modified_dataset=pd.DataFrame({ 'Shelves':newshelves,
    'Clabels':WDClabels,
    'ID':WDitemid})

modified_dataset.to_csv('SVM_dataset.csv',sep='\t', encoding='utf-8')


