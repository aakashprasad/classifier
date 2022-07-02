import numpy as np
import matplotlib.pyplot as plt
import random
import csv

#loading data from csv and creating necessary matrices
j=0
dataMatrix=np.ones((100,3))
feature_set=np.ones((100,3))
labels=np.ones((100,1))
file=open("data.csv")
data=csv.reader(file)
for row in data:
    dataMatrix[j]=row
    j+=1
file.close()
for i in range(0,2):
    feature_set[:,i]=dataMatrix[:,i]
feature_set[:,2]=feature_set[:,0]*feature_set[:,1]
labels=dataMatrix[:,2]
labels=labels.reshape((100,1))

#feature scaling
m=len(feature_set)
avg=(1/m)*np.sum(feature_set,axis=0)
rng=np.amax(feature_set,axis=0)-np.amin(feature_set,axis=0)
feature_set=(feature_set-avg)/rng
feature_set=np.hstack((np.ones((100,1)),feature_set))
feature_set=feature_set.reshape((100,4))
iterNo=200
cost_history=np.ones((iterNo,2))

#sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#gradient descent
def gradient_descent():
    global theta
    random.seed(42)
    theta = random.random()*np.ones((4, 1))
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

#predicts output based on the given input
def predict():
    global student
    global output
    x1=int(input("Enter the marks for exam 1"))
    x2=int(input("Enter the marks for exam 2"))
    x3=x1*x2

    student=np.array([1,(x1-avg[0])/int(rng[0]),(x2-avg[1])/int(rng[1]),(x3-avg[2])/int(rng[2])])
    output=sigmoid(np.dot(student,theta))
    print("Probability of Passing:",float(output)*100,"%")
    if output>=0.5:
        print("Will Pass")
    else:
        print("Will Not Pass")

#produces required graphs of hypothesis, cost vs iteration and decision boundary
def plotGraphs():
    x=np.linspace(-0.5,0.5,100)
    x=x.reshape((100,1))
    y=x.reshape((100,1))
    for j in range(1,4):
         x=np.hstack((y,x))
         j+=1
    x=x.reshape((100,4))
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
        if newFeature_set[i,4]==0:
            plt.scatter(newFeature_set[i,1],newFeature_set[i,2],color='red')
        else:
            plt.scatter(newFeature_set[i,1],newFeature_set[i,2],color='green')

    x=np.linspace(-0.5,0.5,100)
    y=-(theta[0]+theta[1]*x)/(theta[2]+theta[3]*x)
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.plot(x,y)
    plt.scatter(student[1],student[2],marker='x')
    plt.show()

gradient_descent()
predict()
plotGraphs()
