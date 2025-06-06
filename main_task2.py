import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve
import statsmodels.api 
from statsmodels.api import Poisson
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.stats import gamma
import warnings
warnings.filterwarnings('ignore')

traindata=pd.read_csv("ATLAS-data.csv")
testdata=pd.read_csv("ATLAS-labels.csv")


plt.figure(figsize=(30,20))
plt.plot(traindata)           

corr=traindata.corr()
sns.heatmap(corr)

plt.figure(figsize=(50,50))
for i,col in enumerate(traindata,1):
    sns.boxplot(x=col,data=traindata)
    plt.subplot(6,5,i)

scale=MinMaxScaler()
scaled_data=pd.DataFrame(scale.fit_transform(traindata))

x_train,x_test,y_train,y_test=train_test_split(scaled_data,testdata,test_size=0.30,random_state=111)

"""## NAIVE BAYES
Naive Bayes is the most straightforward and fast classification algorithm, which is suitable for a large chunk of data.
Spam filtering, text classification, sentiment analysis, and recommender systems have all employed the Naive Bayes classifier successfully. It predicts an unknown class using Bayes' theorem of probability.

    P(A|B) = P(B|A) * P(A) / P(B).

Naive Bayes classifier calculates membership probabilities for each class, such as the likelihood that a certain record or data point belongs to that class. The most likely class is defined as the one having the highest probability."""


NB=GaussianNB()
NBModel=NB.fit(x_train,y_train)

NB_predict=NBModel.predict(x_test) 

"""### Evaluation
Metrics like confusion_matrix,classification_report,accuracy_score,roc_score are used to evaluate the model.\
confusion_matrix : A confusion matrix is a technique for summarizing the performance of a classification algorithm.\
calssification_repot : It evaluates the model and gives Precision, Recall and F-1 score of the model.\
accuracy_score : Accuracy is a metric used for evaluating classification model. Informally, accuracy refers to the percentage of correct predictions made by our model."""

print('NaiveBayes accuracy:', accuracy_score(y_test,NB_predict))
print('NaiveBayes:', classification_report(y_test,NB_predict))
print('NaiveBayes:', confusion_matrix(y_test,NB_predict))

"""### ROC_curve
ROC is known as Reciever Operating Characteristics curve. It is used for evaluating the performance of a classification model. It compares True positive rate vs False positive rate."""


fpr_NB,tpr_NB,threshold=roc_curve(y_test,NB_predict)
plt.title("ROC")
plt.plot(fpr_NB,tpr_NB)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


"""# Logistic Regression
Logistic regression are the calssification problems with two outputs.The logistic regression model squeezes the output of a linear equation between 0 and 1 using the logistic function.

    log(p/1-p)"""

LR=LogisticRegression()
LRM = LR.fit(x_train,y_train)

LR_predict = LRM.predict(x_test)

print('LogisticRegression:',accuracy_score(y_test,LR_predict))
print('LogisticRegression:',classification_report(y_test,LR_predict))
print('CM:',confusion_matrix(y_test,LR_predict))

### ROC Curve
fpr_LR,tpr_LR,threshold=roc_curve(y_test,LR_predict)
plt.title("ROC")
plt.plot(fpr_LR,tpr_LR)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Comparing two models
plt.title('ROC comparision')
plt.plot(fpr_NB, tpr_NB, 'b', label = "Naive Bayes")# ROC  of Naive Bayes
plt.plot(fpr_LR, tpr_LR, color='green',label='Logistic Regression')# ROC of Logistic Regression
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# conclusions 
print("accuracy of Logistic Regression:",accuracy_score(y_test,LR_predict))
print("accuracy of Naive Bayes:",accuracy_score(y_test,NB_predict))
print('LogisticRegression:',classification_report(y_test,LR_predict))
print('NaiveBayes:',classification_report(y_test,NB_predict))
