import os
import pandas as pd
import seaborn as sns
import collections
from itertools import chain
from mlxtend.frequent_patterns import apriori #import package
from mlxtend.frequent_patterns import association_rules

def changeDir(directory):
    return os.chdir(directory)

changeDir("./Downloads")#change to you own directory

def read_data(data):
    return pd.read_csv(data)

dataset = read_data("binarized_data.csv") #read data from current directory
df = pd.DataFrame(dataset.T, columns = list(range(1,len(dataset.index-1)))) #transpose the data in order to let the algorithm to consider each neurons as individual

#Checking missing values
df.isnull().values.any()

#No need to split data because we do not need build models here

#Histograms plot
lst = dataset.values.tolist()
p = list(chain(*lst)) #flatten all data

sns.set(rc={'figure.figsize':(11.7, 8.27)}) #Histogram plot of spiking or not
sns.distplot(p,kde=False).set_title('Frequency of 1 and 0 (Spiking or Not)')

counter=collections.Counter(p) #Count freq of spiking or not
print(counter)

#Sparse data
#Basic statistics
dat = dataset
idx = dat.columns.values[0:]
stat = pd.DataFrame()

stat['sum'] = dat[idx].sum(axis=1)
stat['min'] = dat[idx].min(axis=1)
stat['max'] = dat[idx].max(axis=1)
stat['mean'] = dat[idx].mean(axis=1)
stat['std'] = dat[idx].std(axis=1)

stat.sort_values("max", ascending=[0]) #Sort the statistics by max values

def freq(min_support = 0.00001, use_columns = True, max_len = None, n_jobs = 2): #parameters can be changed
    freq = apriori(df, min_support, use_columns, max_len, n_jobs) #set minimum support as 0.00001 to extract max number of itemset.
                                                                                            #use the default column names
    return freq

def association_rule(freq, metric = 'confidence', min_threshold = 0.0001, support_only = False): #metric: confidence, support, lift, leverage, conviction
    assoc = association_rules(freq, metric, min_threshold, support_only) #calculate the association_rules by frequent_patterns found above
    sortVal = assoc.sort_values("confidence", ascending = [0])  #sort the association rules by confidence
    print(sortVal)


association_rule(freq())
