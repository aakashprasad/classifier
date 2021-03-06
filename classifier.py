import numpy as np
import matplotlib.pyplot as plt
import random
import csv

j=0
dataMatrix=np.ones((100,3))
feature_set=np.ones((100,2))
labels=np.ones((100,1))
file=open("data.csv")
data=csv.reader(file)
for row in data:
    dataMatrix[j]=row
    j+=1
file.close()
for i in range(0,2):
    feature_set[:,i]=dataMatrix[:,i]
labels=dataMatrix[:,2]
labels=labels.reshape((100,1))

m=len(feature_set)
avg=(1/m)*np.sum(feature_set,axis=0)
rng=np.amax(feature_set,axis=0)-np.amin(feature_set,axis=0)
feature_set=(feature_set-avg)/rng
feature_set=np.hstack((np.ones((100,1)),feature_set))
feature_set=feature_set.reshape((100,3))
iterNo=2000
cost_history=np.ones((iterNo,2))


def sigmoid(x):
    return 1/(1+np.exp(-x))
def gradient_descent():
    global theta
    random.seed(42)
    theta = random.random()*np.ones((3, 1))
    learning_rate=0.9
    for i in range(1,iterNo):
        hypothesis = sigmoid(np.dot(feature_set, theta))
        error=hypothesis-labels
        cost=(-np.dot(labels.transpose(),np.log(hypothesis))-np.dot((1-labels).transpose(),np.log(1-hypothesis)))/m
        cost_history[i,0]=i
        cost_history[i,1]=cost
        theta=theta-(learning_rate/m)*(np.dot(np.transpose(feature_set),error))
    print("Theta is:",theta)
    print("Cost is:",cost)
def predict():
    global student
    global output
    x1=int(input("Enter the marks for exam 1"))
    x2=int(input("Enter the marks for exam 2"))
    student=np.array([1,(x1-avg[0])/int(rng[0]),(x2-avg[1])/int(rng[1])])
    output=sigmoid(np.dot(student,theta))
    print("Probability of Passing:",float(output)*100,"%")
    if output>=0.5:
        print("Will Pass")
    else:
        print("Will Not Pass")
def plotGraphs():
    x=np.linspace(-0.5,0.5,100)
    x=x.reshape((100,1))
    y=x.reshape((100,1))
    for j in range(1,3):
         x=np.hstack((y,x))
         j+=1
    x=x.reshape((100,3))
    plt.title("Hypothesis")
    plt.plot(np.dot(x,theta),sigmoid(np.dot(x,theta)))
    plt.scatter(np.dot(feature_set,theta),labels,marker='X')
    plt.scatter(np.dot(student,theta),output,marker='X',color='red')
    plt.show()

    plt.title("Cost vs Iteration")
    plt.plot(cost_history[:,0],cost_history[:,1])
    plt.show()

    plt.title("Decision Boundary")
    newFeature_set=np.hstack((feature_set,labels))
    for i in range(0,100):
        if newFeature_set[i,3]==0:
            plt.scatter(newFeature_set[i,1],newFeature_set[i,2],color='red')
        else:
            plt.scatter(newFeature_set[i,1],newFeature_set[i,2],color='green')

    x=np.linspace(-0.5,0.5,100)
    y=-(theta[0]+theta[1]*x)/theta[2]
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.plot(x,y)
    plt.scatter(student[1],student[2],marker='x')
    plt.show()

gradient_descent()
predict()
plotGraphs()
