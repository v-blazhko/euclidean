import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
import time
from numpy import array
from matplotlib.pyplot import ion

def unpickle(file):
#Load byte data from file
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
 percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
 filledLength = int(length * iteration // total)
 bar = fill * filledLength + '-' * (length - filledLength)
 print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
# Print New Line on Complete
 if iteration == total: 
  print()

def load_cifar10_data(data_dir):
# Return train_data, train_labels, test_data, test_labels
# The shape of data is 32 x 32 x3
 train_data = None
 train_labels = []

 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic['data']
  else:
   train_data = np.vstack((train_data, data_dic['data']))
  train_labels += data_dic['labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic['data']
 test_labels = test_data_dic['labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 train_labels = np.array(train_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 test_labels = np.array(test_labels)

 return train_data, train_labels, test_data, test_labels

# Computes the classification accuracy for predicted labels _pred_ as compared to the ground truth labels _gt_
def cifar_10_evaluate(pred,gt):
 indexes = 0
 for i, val in enumerate(pred):
  if pred[i] == gt[i]:
   indexes = indexes+1
# Counting number of predicted labels, which are same as true labels (how many)
 l = indexes
 p = l/len(gt)*100
 print('The classification accuracy is '+  str(p) + "%")
 return p


def cifar_10_rand(x):
 pred = []
 for i in range(len(x)):
# Here we generate random number and append it to pred
  numb = np.int32(random.randint(0,9))
  pred.append(numb)
 x = x.tolist()
 cifar_10_evaluate(pred, x)
 print ('< -- for random')


def cifar_show_guess(img,lbl_pred,lbl_real):
# In order to check where the data shows an image correctly
 ion()
 plt.imshow(img)
 plt.title(label_names[lbl_real] + ' - real; predicted: ' + label_names[lbl_pred])
 plt.show()
 plt.pause(2)
 plt.close()


def cifar_10_1NN(test_data, test_labels, train_data, train_labels):
 pred = []
 train_labels = train_labels.tolist()
 test_labels = test_labels.tolist()
 i = 0
 l = len(test_data)
 # Initial call to print 0% progress
 printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete')

# For each element of test_data calculate Euclidean distance to each element of train_data
# For the least distance choose it's label from train_labels. Then compare with test_labels
 for test_item in test_data:
  distances = []
  for train_item in train_data:   
   distances.append(np.linalg.norm(test_item - train_item))
# Then we choose the best result
  minin,indx = min((distances[i],i) for i in range(len(distances)))
  printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete')
  pred.append(train_labels[indx])
  cifar_show_guess(test_item, train_labels[indx], test_labels[i])
  i = i + 1  
 cifar_10_evaluate(pred, test_labels)
  


# Main code  
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_dir = 'cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

cifar_10_rand(test_labels)

# Set up the amount of pictures to be processed, up to 10000
cifar_10_1NN(test_data[0:100], test_labels[0:100], train_data, train_labels)