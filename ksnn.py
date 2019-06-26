import math
from sklearn.model_selection import train_test_split
from sklearn import svm
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
import operator





def euclideanDistance(instance1, instance2,length ):#length- no of features you want to consider for distance
    distance=0
    for x in range(length):
        distance+=pow(abs(instance1[x]- instance2[x]),2)
    return math.sqrt(distance)





def getDistances(trainingSe): #This will return the distance matrix (nxn) 
    distances=[[0 for i in range(len(trainingSe))] for j in range(len(trainingSe))]
    for i in range(len(trainingSe)):
        for j in range(len(trainingSe)):
            distances[i][j]=euclideanDistance(trainingSe[i],trainingSe[j],len(trainingSe[0]))
    return distances



kbar=0
Kc=2
minpts=5
dataset = np.loadtxt("C:/Users/GARVIT/Downloads/gp.csv", delimiter = ',')#loading the dataset and separating features from class labels into trainset and Y
trainSet=[]
trai=[]# copy of trainset
Y=[]
for data in dataset:
    arr=[]
    for val in data[1:]:
        arr.append(val)
    trainSet.append(arr)
    trai.append(arr)
    fx = int(data[0])
    if(fx==2):
        Y.append(-1)
    else:
        Y.append(1)
print(trainSet)
print(Y)



#trainSet=[[9,2],[2,9],[1,7],[1,8],[7,1],[8,1]]#dataset considered for example here.These are coordinate points in 2 axes graph.
k=10#No of nearst points we want to consider
g=0# maximum no of loop iteration allowed
while((g!=1) and (kbar<Kc)):#kbar is basically Kdash and K is no of classes we want. Kdash is no of classes we get after preclustering of the dataset and is initialised 0
    distmat=getDistances(trainSet)#distance between each data point nxn matrix
    a=getDistances(trainSet)#distance between each data point nxn matrix(copy of distmat)
    #print(a)
    s=[[0 for i in range(k)]for j in range(len(trainSet))]
    sor=list(distmat)#sorted distmat
    for i in range(len(trainSet)):
           sor[i].sort()
    #print(sor)
    for i in range(len(trainSet)):
        for j in range(k):
            s[i][j]=sor[i][j+1]# s- it stores distances of first k nearest neighbours of each data points nxk matrix
    #print(s)
    snn=[[0 for i in range(k)] for j in range(len(trainSet))]
    temp=0
    i=0
    for i in range(len(trainSet)):# for finding the coordinates from the distances in s matrix from distance matrix
        j=0
        for j in range(k):
            m=0
            for m in range(len(trainSet)):
                if((s[i][j])==(a[i][m])):
                    temp=m
                    break
            snn[i][j]=trainSet[temp]
    #print(snn)# coordinates of first k nearest neighbours of each point from s matrix(nxk matrix)
    count=[[0 for i in range(len(trainSet))]for j in range(len(trainSet))]# it stores no of shared ponts between each pair of data poins nxn matrix
    for i in range(len(trainSet)):
        for j in range(len(trainSet)):
            c=0
            for n in range(k):
                for m in range(k):
                    if(snn[i][n]==snn[j][m]):
                        c=c+1
            if(i==j):
                count[i][j]=0
            else:
                count[i][j]=c
    print(count)
    cla=1
    clas=[0 for i in range(len(trainSet))] #for storing the classes assigned to each data point n length array
    copy=[0 for i in range(len(trainSet))]# copy of clas ex output[1,2,2,2,1,1]
    for i in range(len(trainSet)):
        for j in range(len(trainSet)):
            if(count[i][j]>=minpts):
                if(clas[i]==0&clas[j]==0):
                    clas[i]=cla
                    copy[i]=cla
                    clas[j]=cla
                    copy[j]=cla
                    cla=cla+1
                else:
                    if(clas[i]!=0):
                        clas[j]=clas[i]
                        copy[j]=clas[i]
                    else:
                        clas[i]=clas[j]
                        copy[i]=clas[j]
    #print(clas)
    kbar=max(clas)
    no=0#storing no of points in first K no of classes from Kdash classes obtained aftr preclustering
    clas.sort()
    i=0
    leng=len(clas)
    while(i<leng): 
        if(clas[i]<=Kc):
            no=no+1
        i=i+1
    if(kbar<Kc):# if Kdash classes less than K classes required then increase min points
        minpts=minpts+1
    elif(no<(0.6*leng)):
        k=k+1
        kbar=0
    else: break
    g=g+1
print(copy)
#print(kbar)
cm=[[0 for i in range(2)]for j in range(len(trainSet))]
print(cm)
for i in range(len(trainSet)):
     for j in range(1):
            cm[i][0]=i
            cm[i][1]=copy[i]# index f each data point and its class label
print(cm)
countcl=[0 for i in range(kbar+1)]
for i in range(len(trainSet)): 
    countcl[copy[i]]=countcl[copy[i]]+1 #printing total number of data points in each class
print(countcl)

for i in range(len(trainSet)):
    if(copy[i]>Kc):
        trainSet.pop(i)#Remving data points where class label>Kc and also removing its class label from the dataset
        Y.pop(i)
print(len(trainSet))



w=2
cmem=0
memfunc=[[0 for i in range(Kc+1)]for j in range(len(trai))]
for i in range(len(count)):
    cmem=0
    cou=[0 for p in range(Kc+1)]
    for j in range(len(count[0])):#Membership value of each point wrt class
        if(count[i][j]>=w):
            cmem=cmem+1
            if(copy[j]==copy[i]):
                cou[copy[j]-1]=cou[copy[j]-1]+1
            else:
                cou[copy[j]-1]=cou[copy[j]-1]+1
    #print(cou)
    for m in range(Kc+1):
        memfunc[i][m]=cou[m]/cmem
print(memfunc)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainSet, Y, test_size=0.3)
Xtrain = np.asarray(Xtrain)
Ytrain = np.asarray(Ytrain) 
clf = svm.OneClassSVM(nu=0.99, kernel="rbf", gamma=2)#Applying OCSVM over remainig data points
clf.fit(Xtrain)
y_pred_train = clf.predict(Xtrain)
y_pred_test = clf.predict(Xtest)
print(accuracy_score(Ytrain, y_pred_train))
print(accuracy_score(Ytest, y_pred_test))
               
