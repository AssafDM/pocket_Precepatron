from time import sleep
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm




#in this project i implemented a percepatron pocket algorithm to differentiate between two hand written digit using the mnist db.
#ichose the an algebric approach that utilizes linear algebra suloutions implemented by numpy to improve efficiency.
#the machine learning is done by multiplying the data matrix with the weights+bias vector to recieve a prediction vector.
#the prediction vector is then compered to the true labels vector to recieve an addition map vector according to the percepatron algorithm.
#the addition map is moltiplied by the data matrix to recieve a sum vector of the addition and substraction mentioned. 
#this sum vector is added to the current weight vector to create an updated vector. 
#the pocket mechanism works by checking the accuracy each time and updating the pocket if the accuracy is getting better. 
#
#at the end of the run of the main function it analyses and print stats about the run.


class Perceptron:
    def __init__ (self, X,y ) :
        self.count, self.n = X.shape
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=316527282)
        self.w= np.zeros((self.n+1,1)) #set w vector + bias to 0
        self.p= np.zeros((self.n+1,1)) #set w vector + bias to 0
        self.p_perc=0
        self.X_train = np.hstack((np.ones((self.X_train.shape[0],1),float), self.X_train))  #set x0 to 1, x1-xn to data
        self.X_test = np.hstack((np.ones((self.X_test.shape[0],1),float), self.X_test))  #set x0 to 1, x1-xn to data

    def train(self):
      #print('training')
      result = np.sign(self.X_train @ self.w).T


      #calculate success
      percentage= self.accuracy(self.y_train, result)
      #print("this rounds W had", (int)(percentage*100), "% success")
      #case better than pocket: save w in pocket
      if percentage>self.p_perc:
        self.p=self.w
        self.p_perc = percentage
      #update w
      d= np.sign(self.y_train -result) #calculate the addition map
      addvec=  d @ self.X_train #sum up all additions
      self.w= self.w + addvec.T #add to w vector
      return result

    def test(self):
      result = np.sign(self.X_test @ self.p).T #calculate resukt and return result vector
      return result

    def accuracy(self, y_true, y_pred):
      d=(y_true == y_pred).astype(int)    #create vector mask where two vectors are similar (AND operation)
      percentage= (float(d.sum()) / y_true.shape[0]) #dvide true predivtions by total amount to get accuracy
      return percentage

    def success_analysis(self, y_true, y_pred):
      y_pred= y_pred.T
      tp, tn, fp,fn =0,0,0,0
      for i in range(y_true.shape[0]):
        #positive
        if y_true[i]==1:
          if y_pred[i]==1: tp+=1
          else: fn+=1
        #negative
        elif y_pred[i]==-1: tn+=1
        else: fp+=1
      res = [tp,tn,fp,fn]
      return res

def initialize():

  #Access features (pixel values)
  mnist = fetch_openml('mnist_784', version=1, cache=True, parser = 'auto')
  X=mnist['data']
  X['target'] = mnist['target'] #bring in the labels
  # Filter the dataset together to include only samples corresponding to digits 2 and 7
  X = X[(X['target'] == '2') | (X['target'] == '7')]
  y = X['target'] #extract labels
  X=X.drop(columns=['target']).astype(float).values*(1.0/255.0) # standardise data
  #make y to contain -1,1 only
  y= y.values.astype(int)
  y = np.where(y == y.min(), -1, y)
  y = np.where(y >=0, 1, y)
  y= np.array(y)

  p = Perceptron(X,y)
  return p

if __name__ =="__main__":
  p= initialize() #initialize precepatron
  #set data plotting lists
  iteration = []
  trainloss = []
  testloss = []
  #run training and updat list,
  #i chose 300 epochs because the alg. stabilizes quickley and setteles,
  #then it seems to be starting abnormal behavior around epoch 200, i believe this is due to linear dependency of some vectors
  for i in tqdm(range(0,300)):
    if p.p_perc ==1:
      break
    temp = p.train()
    train =1-p.accuracy(p.y_train ,temp)
    test =  1-p.accuracy(p.y_test , p.test())
    iteration.append(i+1)
    trainloss.append(train)
    testloss.append(test)
  success = p.success_analysis(p.y_test , p.test())
  print(f"my models final accuracy is: {(1-test)*100}%\nwith best performance to peak at: {(1-min(testloss))*100}% in iteration {testloss.index(min(testloss))+1}")
  print(f"the succes slicing is:\nTP:\t{success[0]}\nTN:\t{success[1]}\nFP:\t{success[2]}\nFN:\t{success[3]}")
  print(f"the sensitivity is:\t{success[0]/float(success[0]+success[3])}")
  print(f"the selectivity is:\t{success[1]/float(success[1]+success[2])}")
  print("the following graphs show \n\t(1)loss over iteration\n\t(2)confusion matrix of the final algorithm on unseen data ")
    # Plot data
  plt.plot(iteration, trainloss, label='Train loss', color='blue')
  plt.plot(iteration, testloss, label='Test loss', color='red')
  # Set y-axis limits (magnified)
  plt.ylim(0, 0.07)  # Magnify the y-axis by setting the limits from 0 to 0.07


  # Add labels and title
  plt.xlabel('Iteration')
  plt.ylabel('loss')
  plt.title('Train and Test Loss over Iterations')

  # Add legend
  plt.legend()

  # Show grid
  plt.grid(True)

  # Show plot
  plt.show()
  confmat= confusion_matrix(p.y_test, p.test().T)
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confmat, display_labels = [2, 7])
  cm_display.plot()

