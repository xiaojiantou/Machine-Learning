
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv


# In[2]:


X_train_csv = csv.reader(open('/Users/arrowlittle/Desktop/MachineLearning/hw2-data/X_train.csv'))
X_train = np.array(list(X_train_csv)).astype("float")
X_test_csv = csv.reader(open('/Users/arrowlittle/Desktop/MachineLearning/hw2-data/X_test.csv'))
X_test = np.array(list(X_test_csv)).astype("float")

y_train_csv = csv.reader(open('/Users/arrowlittle/Desktop/MachineLearning/hw2-data/y_train.csv'))
y_train = np.array(list(y_train_csv)).astype("float")
y_test_csv = csv.reader(open('/Users/arrowlittle/Desktop/MachineLearning/hw2-data/y_test.csv'))
y_test = np.array(list(y_test_csv)).astype("float")


# # part a

# In[3]:


B0=[]
B1=[]
def parameter():
    for j in range(54):
        t0=(X_train[:,j].T.dot((1-y_train))/np.ones(4508).T.dot(1-y_train))
        t1=(X_train[:,j].T.dot(y_train)/np.ones(4508).T.dot(y_train))
        B0.append(t0)
        B1.append(t1)
    for j in range(54,57):
        t0=(np.ones(4508).T.dot(1-y_train)/(np.log(X_train[:,j]).T).dot(1-y_train))
        t1=(np.ones(4508).T.dot(y_train)/(np.log(X_train[:,j]).T).dot(y_train))
        B0.append(t0)
        B1.append(t1)
pi=(1/4508)*(np.ones(4508).T.dot(y_train))


# In[4]:


def classifier(B0, B1, i):
    parameter()
    p0=1-pi
    p1=pi
    for j in range(54):
        c0=((B0[j])**(X_test[i,j]))*((1-B0[j])**(1-X_test[i,j]))
        c1=((B1[j])**(X_test[i,j]))*((1-B1[j])**(1-X_test[i,j]))
        p0=p0*c0
        p1=p1*c1
    for j in range(54,57):
        c0=((B0[j])*(X_test[i,j])**(-(B0[j]+1)))
        c1=((B1[j])*(X_test[i,j])**(-(B1[j]+1)))
        p0=p0*c0
        p1=p1*c1
    if p0>p1:
        return 0
    else:
        return 1


# In[5]:


def prediction(B0, B1, i):
    if(classifier(B0,B1,i)==0 and y_test[i]==0):
        return 0
    elif(classifier(B0,B1,i)==0 and y_test[i]==1):
        return 1
    elif(classifier(B0,B1,i)==1 and y_test[i]==0):
        return 2
    elif(classifier(B0,B1,i)==1 and y_test[i]==1):
        return 3


# In[6]:


pre00=0
pre01=0
pre10=0
pre11=0
for i in range(93):
    if(prediction(B0, B1, i)==0):
        pre00+=1
    elif(prediction(B0, B1, i)==1):
        pre01+=1
    elif(prediction(B0, B1, i)==2):
        pre10+=1
    elif(prediction(B0, B1, i)==3):
        pre11+=1
print (pre00, pre01, pre10, pre11)


# In[7]:


def accuracy(B0, B1, pre00, pre01, pre10, pre11):
    correct = pre00+pre11
    wrong = pre01+pre10
    acc = correct/93
    return acc


# In[8]:


print(accuracy(B0, B1, pre00, pre01, pre10, pre11))


# # part b

# ### B0 & B1

# In[9]:


plt.close('all')
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
markerline, stemlines, baseline = ax1.stem(range(1, 55), B0[:54], '-.')
plt.setp(baseline, 'color', 'b', 'linewidth', 2)
plt.setp(markerline, 'markerfacecolor', 'r')
plt.setp(stemlines, 'color', 'r')
markerline, stemlines, baseline = ax1.stem(range(1, 55), B1[:54], '-')
plt.setp(baseline, 'color', 'b', 'linewidth', 2)
plt.setp(markerline, 'markerfacecolor', 'y')
plt.setp(stemlines, 'color', 'g','alpha', 0.8)
ax1.set_xlabel("Dimensions")
ax1.set_ylabel("Bernoulli Parameters")
plt.legend(["Class y=0", "Class y=1"], loc='best', numpoints=2)
plt.title("Plot for Bernoulli Parameters when y = 0 (Red) and y = 1 (Yellow)")
plt.show()


# ### pambase.names file

# In[10]:


MLE0_16 = B0[16]
MLE1_16 = B1[16]
print(MLE0_16, MLE1_16)
MLE0_52 = B0[52]
MLE1_52 = B1[52]
print(MLE0_52, MLE1_52)


# The dimension 16 is for the word frequency of word “free” (word_freq_free). As for Class 0, the MLE estimate is 0.09553441 and for Class 1, the MLE estimate is 0.38344595. So, we can say that for Class 0 (not a spam mail) the likelihood of seeing the word free is 0.09553441 less than Class 1 (spam mail) where it is 0.38344595.
# 
# The dimension 52 is for the char frequency of char “!” (char_freq_!). As for Class 0, the MLE estimate is 0.10468521 and for Class 1, the MLE estimate is 0.61486486.  So, we can say that for Class 0 (not a spam mail) the likelihood of seeing the char “!” is 0.10468521 less than Class 1 (spam mail) where it is 0.61486486.
# 
# In addition, both dimension 16 and dimension 52 are Bernoulli distribution and their parameters are just the average number of documents that have (y = 0 |X = x (free)), (y = 1 |X = x (free)), (y = 0 |X = x (!)), (y = 1 |X = x (!)). In conclusion, spam emails are more likely have word “free” and char “!” than non-spam mails.

# # part c

# In[11]:


def normalize(j, X_t):
    temp = []
    minimum = min(X_t[:,j])
    maximum = max(X_t[:,j])
    difference = maximum - minimum
    for i in range(len(X_t)):
        temp.append(((X_t[i,j])-minimum)/difference)
    return temp


# In[12]:


def preprocess(X_tr,X_te):
    X_train1=X_train.copy()
    X_test1=X_test.copy()
    for j in range(54,57):
        X_train1[:,j] = normalize(j,X_tr)
        X_test1[:,j] = normalize(j,X_te)
    return X_train1, X_test1
X_train1 ,X_test1 = preprocess(X_train,X_test)


# In[13]:


def distance(p, q):
    dis = 0
    for j in range(57):
        dis += np.absolute((X_train1[p,j]-X_test1[q,j])) 
        #print (dis)
    return dis


# In[14]:


kNN_array=[]
def kNN(q):
    dis=[]
    for i in range(4508):
        dis.append(distance(i, q)) 
    idx = np.argsort(dis)
    kNN_array.append(idx[:20])
for q in range(93):
    kNN(q)


# In[15]:


def kNN_prediction(q, k):
    count0=0
    count1=0
    idx=kNN_array[q]
    for n in idx[:k]:
        if(y_train[n]==0):
            count0+=1
        else:
            count1+=1
    if (count0>count1):
        return 0
    else:
        return 1


# In[16]:


def kNN_accuracy(k):
    cor = 0
    for q in range(93):
        if (kNN_prediction(q, k) == y_test[q]):
            cor += 1
    return cor/93


# In[17]:


acc_kNN=[]
for k in range(20):
    acc_kNN.append(kNN_accuracy(k+1))


# In[18]:


plt.close('all')
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
ax1.set_xlabel("the Number of Neighbors")
ax1.set_ylabel("Accuracy")
plt.plot(acc_kNN)
plt.show()


# # part d

# In[19]:


X_train2 = np.hstack((X_train,np.ones((4508,1))))
X_test2 = np.hstack((X_test,np.ones((93,1))))
y_train2 = y_train*2-1
y_test2 = y_test*2-1


# In[20]:


print(X_train2[40,-4:-1])


# In[21]:


print(y_train2.reshape((4,1127)))


# In[22]:


def sigmoid(x):
    t = np.maximum(x,0)
    return np.exp(x-t)/(np.exp(0-t)+np.exp(x-t))


# In[23]:


def largesum(w):
    l=0
    lsum = 0
    for i in range(4508):
        t = (X_train2[i].dot(w))[0]
        s = sigmoid(t*y_train2[i])
        f = (1-s)*y_train2[i]*X_train2[i]
        l = np.add(l,np.log(s+10**(-10))) 
        lsum = np.add(lsum, f) 
    return lsum, l


# In[24]:


def steepest_ascent():
    w = np.zeros((58,1))
    for t in range(10000):
        yita = 1/((10**5)*np.sqrt(t+1))
        lsum,l = largesum(w)
        L.append(l)
        ylsum = yita*lsum.reshape((58,1))
        w = np.add(w,ylsum) 


# In[25]:


L=[]
steepest_ascent()


# In[26]:


plt.close('all')
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
ax1.set_xlabel("Iteration")
ax1.set_ylabel("L")
plt.plot(L)
plt.show()


# # part e

# In[27]:


def derivative(w2):
    l2=0
    w1L=0
    w2L=0
    for i in range(4508):
        t = (X_train2[i].dot(w2))[0]
        s = sigmoid(t*y_train2[i])
        s2 = sigmoid(t)
        l2 = np.add(l2,np.log(s+10**(-10))) 
        f1 = (1-s)*y_train2[i]*X_train2[i]
        f2 = s2*(1-s2)*np.matmul(X_train2[i,:].reshape(58,1),X_train2[i,:].reshape(1,58))
        w1L = np.add(w1L, f1)
        w2L = np.add(w2L, f2)
    #print (w1L.shape, w2L.shape)    
    return w1L, -w2L, l2


# In[28]:


def N_method():
    w2 = np.zeros((58,1))
    for t in range(1,101):
        yita2 = 1/(np.sqrt(t+1))
        w1L, w2L, l2 = derivative(w2)
        w1L = w1L.reshape((58,1))
        #print(w2L.shape)
        w2 = np.subtract (w2, yita2*np.linalg.pinv(w2L).dot(w1L))
        L2.append(l2)
    return w2


# In[29]:


L2=[]


# In[30]:


plt.close('all')
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
ax1.set_xlabel("Iteration")
ax1.set_ylabel("L")
plt.plot(L2)
plt.show()


# In[31]:


def logistic_accuracy():
    cor = 0
    w2 = N_method()
    for i in range(93):
        t = X_test2[i].dot(w2)
        if (sigmoid((t)[0])>0.5 and y_test2[i]==1):
            cor+=1 
        elif (sigmoid((X_test2[i].dot(w2))[0])<=0.5 and y_test2[i]==-1):
            cor+=1
    print (cor)
    return cor/93


# In[32]:


logistic_accuracy()

