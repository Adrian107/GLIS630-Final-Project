import os
import pandas as pd
from mlxtend.frequent_patterns import apriori #import package
from mlxtend.frequent_patterns import association_rules

def changeDir(directory):
    return os.chdir(directory)

changeDir("./Downloads")#change to you own directory

def freq(data, min_support = 0.00001, use_columns = True, max_len = None, n_jobs = 2): #parameters can be changed
    dataset = pd.read_csv(data) #read data from current directory
    df = pd.DataFrame(dataset.T, columns = list(range(1,len(dataset.index-1)))) #transpose the data in order to let the algorithm to consider each neurons as individual
    freq = apriori(df, min_support, use_columns, max_len, n_jobs) #set minimum support as 0.00001 to extract max number of itemset.
                                                                                                #use the default column names
    return freq

def association_rule(freq, metric = 'confidence', min_threshold = 0.0001, support_only = False): #metric: confidence, support, lift, leverage, conviction
    assoc = association_rules(freq, metric, min_threshold, support_only) #calculate the association_rules by frequent_patterns found above
    sortVal = assoc.sort_values("confidence", ascending = [0])  #sort the association rules by confidence
    print(sortVal)


association_rule(freq("binarized_data.csv"))
