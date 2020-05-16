# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import scipy
from random import sample
import math
import random

if __name__ == '__main__':
    df = pd.read_csv('AB_test_data.csv')
    df = df[df.columns.drop('date')]
    groupA = df.loc[df['Variant'] == 'A']
    groupB = df.loc[df['Variant'] == 'B']

    norm = scipy.stats.norm()

    A_list = list(groupA['purchase_TF'])
    B_list = list(groupB['purchase_TF'])
    t_test(A_list,B_list,0.95)
    optimal_size = sample_size_estimation(0.95,0.8,A_list,B_list)
    print('Optimal Size is:',optimal_size)
    random_t_test(A_list,B_list,10,optimal_size,0.95)
    print('The average number of observations in test is {}'.format(math.ceil(sequential_t_test(A_list,B_list,10,optimal_size,0.05,0.8))))

# Assuming equal variances
# Assuming one-tailed
# This function gives us the t-stats
#@param: confidence: the accepted confidence level
#        A: the control group (as an 1-D list)
#        B: the treatment group(as an 1-D list)
#@return: the t-stats
def t_test(A,B,confidence):
    x1 = np.mean(A)
    x2 = np.mean(B)
    s = 0
    for i in range(len(A)):
        s += (A[i]-x1)**2
    for j in range(len(B)):
        s += (B[j]-x2)**2
    s = np.sqrt(s/(len(A)+len(B)-2))
    t = (x2-x1)/(s*np.sqrt(1/len(A)+1/len(B)))
    print('t score is: ', t)
    return t



# assume equal sample size
# Assuming one-tailed
# This function gives us the optimal sample size, given a caertain confidence level and power
#@param: confidence: the accepted confidence level
#        power: power
#        A: the control group (as an 1-D list)
#        B: the treatment group (as an 1-D list)
#@return: the minimal sample size for each group
def sample_size_estimation(confidence,power,A,B):
    x1 = np.mean(A)
    x2 = np.mean(B)
    var1 = np.var(A)
    var2 = np.var(B)
    mde = norm.ppf(power)*np.sqrt(var1/len(A)+var2/len(B))+(x1-x2)
    p = (np.sum(A)+np.sum(B))/(len(A)+len(B))
    n = (((norm.ppf(confidence)*np.sqrt(2*p*(1-p)) + norm.ppf(power)*np.sqrt(x1*(1-x1)+x2*(1-x2))))**2)/(mde**2)
    return n





# assume equal sample size
# Assuming one-tailed
# This function gives us the t test based on random sampling
#@param:  confidence: the accepted confidence level
#         iterations: the number of random sampling 
#         size: sample size
#         A: the control group (as an 1-D list)
#         B: the treatment group (as an 1-D list)
#@return: none
def random_t_test(A,B,iterations,size,confidence):
    size  = math.ceil(size)
    for i in range(iterations):
        A_sample = sample(A,size)
        B_sample = sample(B,size)
        t_test(A_sample,B_sample,confidence)
    return





# This function does the squential test
#@param: iterations: the number of random sampling 
#        size: sample size
#        A: the control group (as an 1-D list)
#        B: the treatment group (as an 1-D list)
#        alpha: 1-confidence interval
#        power: power
#@return: average number of iterations
def sequential_t_test(A,B,iterations,size,alpha,power):
    iteration = list() 
    size  = math.ceil(size)
    for k in range(iterations):
        A_sample = sample(A,size)
        B_sample = sample(B,size)
        ln_A = np.log(1/alpha)
        ln_B = np.log(1-power)
        total_sample = A_sample.copy()
        total_sample.extend(B_sample)
        random.shuffle(total_sample)
        p_A = np.mean(A_sample)
        p_B = np.mean(B_sample)
        accumulative_log_lamda = 0
        i=0
        while (accumulative_log_lamda>ln_B) and (accumulative_log_lamda <ln_A):
            if total_sample[i]:
                accumulative_log_lamda += np.log(p_A/p_B)
            else:
                accumulative_log_lamda += np.log((1-p_A)/(1-p_B))
            i+=1
            if i >= len(total_sample):
                print('Cannot reject or accept H0')
                break
        if accumulative_log_lamda <= ln_B:
            print('Accept H0 in {} trials'.format(i))
        elif accumulative_log_lamda >= ln_A:
            print('Reject H0 in {} trials'.format(i))
        iteration.append(i)
    return np.average(iteration)

