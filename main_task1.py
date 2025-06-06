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

"""The data given has the count of words in each sentence from the a book. To perform analysis on this kind of data we need to 
convert it into the list form which is done in the above cell.

 Poisson Distribution
 The Poisson distribution is used to model the number of events occurring within a given time interval.
 The formula for the Poisson probability mass function is

 p(x; 𝜆  ))= (𝜆^x∗𝑒𝑥𝑝^−𝜆 )/x! for x=0,1,2,⋯

 λ is a shape parameter that represents the average number of events in a certain time interval.
 x is the data point

 ### Maximum Likelihood inference in Poisson
Maximum likelihood estimate is a method for determining the values of a model's parameters. The parameter values are chosen to maximize the likelihood that the model's described process created the data that were correctly predicted.

𝐿(𝜇|𝑥)=𝑝(𝑥|𝜇)

The model parameter L(𝜇|x) is treated as a function that gives us the likelihood of the observed data points.

Based on a statistic model with parameter value, 𝑝(𝑥|𝜇) is the probability density of the observed data x.

"""

data=pd.read_csv("QuietDonSentenceLengths.csv")

data=data.rename(columns={'6': 'Sent_length'})
nrow=pd.DataFrame({"Sent_length": 6},index=[3])
data=pd.concat([nrow, data.iloc[0:]]).reset_index(drop=True)
listOf_Sent_length= data["Sent_length"].values.tolist()# converting data into list


# Maximum likelihood inference in poisson
def LogLikelihoodPoisson(val):
    Loglikelihood = np.sum(poisson.logpmf(listOf_Sent_length,val,loc=0))
    return -Loglikelihood

"""
For the supplied parameter val, the above function provides the products of the individual probabilities densities in logarithmic form of 
each individual data point in listOf_Sent_length.
"""


LogLikelihoodPoisson(7)
minimize(LogLikelihoodPoisson,8)

"""Here we obtained the maximum likelihood estimate from a range of random values taken. Maximization is performed to search over the parameter space.
The best parameter is chosen by minimize function, in the minimize function we try to maximize the loglikelihood."""

minimize(LogLikelihoodPoisson,8).x

PoissonEstimate_MLE=minimize(LogLikelihoodPoisson,8).x

meanval=np.sum(listOf_Sent_length)/len(listOf_Sent_length)

"""The next step involves plotting a graph to show the change in likelihood with the change in the average values. 
we get a maximum likelihood estimate which gives the best paramter of the data."""

RandomVal = np.linspace(5,25,1000) # creating random 1000 values from 5 to 25.

Y_values = [] # Empty list for y coordinates
for n in RandomVal: # for each value of n in the range of random values created
    Y_values.append(-LogLikelihoodPoisson(n)) # adding LogLikelihoodPoisson(n) to the list of y coordinates

plt.plot(RandomVal,Y_values) # plot line for the negative log likelihood
plt.axvline(meanval,color='red') # add a red vertical line at the ML estimate of n
plt.xlabel("Parameter range for poisson distribution")
plt.ylabel("Estimates of Likelihood for each parameter")

"""### Negative Binomial distribution
The negative binomial distribution is also known as the Pascal distribution.
In a negative binomial experiment, a negative binomial random variable is the number X of various tests required to produce r successes. A negative binomial distribution is the probability distribution of a negative binomial random variable.
 A Negative Binomial Distribution generally has high variance than mean, It can be defined  by two parameters mean and shape parameter(k)"""

y = listOf_Sent_length
X = np.ones_like(listOf_Sent_length)
result = statsmodels.api.NegativeBinomial(y, X).fit()
print(result.summary())
print('Parameters: ', result.params)

"""For over-dispersed count data, where the conditional variance exceeds the conditional mean, negative binomial regression can be employed. Because it has the same mean structure as Poisson regression and an extra parameter to describe over-dispersion, it can be regarded a generalisation of Poisson regression.
The confidence intervals for Negative binomial regression are likely to be lower than those for a Poisson regression model if the conditional distribution of the outcome variable is over-dispersed."""


print("The beta and alpha coefficients computed using Negative Binomial Regression Model are as follows ",
      round(result.params[0],3),round(result.params[1],3))
print("The Log Likelihood value:",-result.llf)
print("The negative binomial data's mean or average is determined as", np.exp(result.params[0]))

"""The maximization will be done with the library function, which will find the best parameters for us based on the alpha and beta values acquired from the negative binomial regression findings."""

Alp= result.params[1]
Alp

# Maximum likelihood inference in negative binomial regression
from scipy.stats import nbinom
def NBLikelihood(val):
    num = 1/Alp
    s = num/(num+val)
    loglikelihood = (nbinom.logpmf(listOf_Sent_length, num, s))
    return -np.sum(loglikelihood)


"""The shape parameters for binom are num and s, where num is the number of successes and  is the chance of a single success. The data's mean and alpha can be used to compute num and s. By assuming the alpha value we are trying to figure out what the mean is."""


NBLikelihood(8)

minimize(NBLikelihood,8)

minlikelihood=minimize(NBLikelihood,8).x

"""In the above step we find the maximum likelihood estimate by using minimize function to the loglikelihood value."""

RandVal = np.linspace(5,25,1000) # creating an array of 1000 random values from 5 to 25.

Y_cord = [] # Empty list of y coordinates
for n in RandVal: # for each value of n in the range
    Y_cord.append(-NBLikelihood(n)) # add NBLikelihood(n) to list of y coordinates

plt.plot(RandVal,Y_cord) # plotting line for the negative log likelihood
plt.axvline(minlikelihood,color='red') # adding a  vertical line at the ML estimate of n
plt.xlabel("Negative binomial distribution parameter ranges")
plt.ylabel("Likelihood Estimates for each parameter")

"""### AIC
You can use the Akaike Information Criterion (AIC) to see how well your model fits the data set without overfitting it.
Unless it is compared to the AIC score of a competitive model, the AIC score is not of big use.

            AIC = -2(log-likelihood) + 2K
            where k is the number of model parameters"""

###Comparison AIC values.
result1 = statsmodels.api.NegativeBinomial(y, X).fit()
result2 = statsmodels.api.Poisson(y, X).fit()

print("AIC values for poisson regeression",result1.aic)
print("AIC values for negative binomial regeression",result2.aic)

"""### MAP inference or Maximum a posteriori inference
MAP inference finds the parameter value that maximises the posterior distribution.
The parameter value that minimizes the posterior distribution is determined through MAP inference.
Fitting a Poisson distribution to the data with MAP estimation and a Gamma distribution prior on the Poisson distribution's parameter, with the Gamma distribution's shape 4 and scale 0.1.

    P(A|B) = (P(B|A) * P(A))/P(B)
    P(A|B) is proportional to P(B|A) * P(A)
    P(theta|X) = P(X|theta) * P(theta)
    maximize P(X|theta) * P(theta)
We'll do MAP inference by adding a prior to our likelihood (in log form), so that the posterior density is maximized.   """

"""### Posterior
In Bayesian statistics, a posterior probability is the revised or updated likelihood of an event occurring after new information is taken into consideration.
Adding prior density to the log likelihood yields the negative logarithm of the posterior.
The total of distinct log likelihoods for each data point is called loglikelihood. In the gamma distribution, the prior probability density is generated for every val, which corresponds to the probability of an individual point (val) under the given parameter shape and scale."""


def NegLogposterior(val):
    loglikelihood = np.sum(poisson.logpmf(listOf_Sent_length,val,loc=0))
    prior = gamma.logpdf(val,4,loc=0.1,scale=0.1)
    return -(loglikelihood+prior)


 minimize(NegLogposterior,6)

 RandValu = np.linspace(5,15,1000) 
#The majority of data sentence lengths, or the average of data sentence lengths, are between 5 and 30 words..

listE = [] # Eempty list of y coordinates
for n in RandValu: # for each value of mu in the range
    listE.append(-NegLogposterior(n)) # add NegLogposterior(n) to the list of n

plt.plot(RandValu,listE) # The negative log probability is represented by a line.

MapEstimate = minimize(NegLogposterior,6).x

plt.axvline(PoissonEstimate_MLE,color='red') # add a vertical line at the ML estimate of n using poisson distribution
plt.axvline(MapEstimate,color='yellow') # add a green vertical line at the ML estimate using MAP
plt.legend(MapEstimate)


"""The above plot indicates the comparision of MAP estimate and Maximum Likelihood Estimate where yellow line indicates the MAP estimate and red line indicates the Maximum Likelihood estimate of poisson distribution. The value where the curve meets the map and ML lines are around at 12.64
The map estimate is slightly ahead of ML estimate as the gamma data consisting of average sentence lengths got affecteed by the scale and shape parameteres."""



