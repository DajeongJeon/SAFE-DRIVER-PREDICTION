import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter

import os
os.chdir("C:/CSC529/finalproject/")

import sys
sys.path.append('C:/python_class/')
from clfFunction import svmclf as scf

np.set_printoptions(precision = 6)



def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# =============================================================================
# 1. Data import
# =============================================================================
# input the original dataset
#toxic_train = pd.read_csv('train.csv', sep=',').fillna(' ')
safedriver = pd.read_csv('cleaned.csv', sep=',')
safedriver.drop(['Unnamed: 0'], axis=1, inplace=True)
safedriver.name = 'safedriver'

safedriver.head(10)
safedriver.shape
safedriver.columns

Counter(safedriver.dtypes.values) # check variables types
safedriver_float = safedriver.select_dtypes(include=['float64']) # float type
safedriver_int = safedriver.select_dtypes(include=['int64']) # int type

# get class names (59)
class_names = safedriver.columns

datalist = [safedriver]

# =============================================================================
# 2.Data exploratory - missing values, feature visualization and Correlation
# =============================================================================
#### fill '0' ####
# count the 'null' value (total and by columns)
for i in datalist:
    print()
    print('Dataset Name: %s' %i.name)
    scf.Missing_value_count(i)

for i in datalist:
    print()
    print('Dataset Name: %s' %i.name)
    print(i.apply(lambda x: x.count(), axis=0))
    
# fill the missing values with '0'
for i in datalist:
    i.fillna(0 ,inplace = True)
    
# check which columns have missing values
# Values of -1 indicate that the feature was missing from the observation. 
missval_df = safedriver[safedriver==-1].count()
missval_df = missval_df.loc[missval_df != 0] # select the features that have missing values
missval_df.sum()/35117508 # total missing value rate in the dataset

# bar chart for missing value (-1)
pos = range(len(missval_df.index))
sns.set_style("whitegrid")
ax = sns.barplot(x=missval_df.values, y=missval_df.index, data=missval_df,color="salmon", saturation=.5)
ax.set(xlabel='Features', ylabel='# of missing values')
ax.set_xticklabels(missval_df.index, rotation='vertical', fontsize=10)
plt.title('# of missing values of features')
plt.show()

safedriver[safedriver==-1].count().sum()
846458/35117508 

#### missing Values ####
# replace missing value (-1) to 'Nan'
for i in datalist:
    i.replace(-1, np.nan, inplace=True)
safedriver[safedriver==-1].count().sum()# check the missing value(-1)
    
# recheck the missing value
for i in datalist:
    print()
    print('Dataset Name: %s' %i.name)
    print(i.apply(lambda x: x.count(), axis=0))

missFeatures = safedriver[safedriver.columns[safedriver.columns.isin(missval_df.index)]]
sns.pairplot(missFeatures.iloc[:,10:])

'ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15'
'ps_reg_01', 'ps_reg_02', 'ps_reg_03'
'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15'

'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',
'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14'

# check # of values for categorical features
cat_features = safedriver.filter(like='_cat')
bin_features = safedriver.filter(like='_bin')
con_ord_features = safedriver.drop(bin_features.columns.values, axis=1).drop(cat_features.columns.values, axis=1).drop(['id','target'], axis=1)

print('Categorical Features: \n {}'.format(cat_features.columns.values))
print('Binary Features: \n {}'.format(bin_features.columns.values))
print('Continuous/Ordinal Features: \n {}'.format(con_ord_features.columns.values))

# check how many values categorical features have
for f in cat_features:
    n = safedriver[f].nunique()
    print("'%s' has %d different value(s)" %(f, n))

pd.DataFrame(cat_features.agg(['min','max']).T, dtype='float') # check min max on features
pd.DataFrame(con_ord_features.agg(['min','max']).T, dtype='float') # check min max on features

# boxplot for Continuous/Ordinal features
#plt.title('Boxplot for Continuous/Ordinal features')
plt.xlabel('Continuous/Ordinal Features')
#plt.ylim([-1, 25])
#con_ord_features.boxplot(rot = 90)
plt.title('Boxplot for Continuous/Ordinal features (exclude ps_car_11_cat)')
con_ord_features.drop('ps_car_11_cat', axis=1).boxplot(rot = 90)
plt.show()

con_ord_features.drop['ps_car_11_cat']


# bar chart for Categorical features
# few of them are almost like Binary (0 or 1)
for f in cat_features:
    plt.title("Bar chart of '{}'".format(f))
    plt.ylabel('Categories')
    cat_features[f].value_counts().plot(kind='barh')
    plt.show()

for f in con_ord_features:
    plt.title("Histogram of '{}'".format(f))
    plt.ylabel('Categories')
    plt.hist(con_ord_features[f])
    plt.show()


# correlation (heatmap) - All features
sns.heatmap(safedriver.iloc[:,1:].corr(), 
            xticklabels=safedriver.iloc[:,1:].columns.values,
            yticklabels=safedriver.iloc[:,1:].columns.values)


# correlation (heatmap) - Continuous/Ordinal features
sns.heatmap(con_ord_features.corr(), 
            xticklabels=con_ord_features.columns.values,
            yticklabels=con_ord_features.columns.values)


# correlation (heatmap) - exclude 'calc' features
calc_list = safedriver.filter(like='_calc')
calc_features = safedriver.drop(calc_list.columns.values, axis=1).drop(['id'], axis=1)

sns.heatmap(calc_features.corr(), 
            xticklabels=calc_features.columns.values,
            yticklabels=calc_features.columns.values)


# correlation (Top 20 correlated fields)
print("Top Absolute Correlations")
get_top_abs_correlations(calc_features, 20)

# Target variable (independent) 
safedriver['target'].value_counts().plot(kind='bar')
safedriver['target'].value_counts()/safedriver.shape[0]


# =============================================================================
# [DT][RF][Train] Raw data 
# =============================================================================

# Divide the data into Trainand test    
safedriver_x = safedriver.drop(['target'], axis=1)
safedriver_y = safedriver['target']
safedriver_x.shape
safedriver_y.shape

x_train,x_test_all,y_train,y_test_all = train_test_split(safedriver_x,safedriver_y,test_size = 0.34,random_state=9)

# =============================================================================
# [Bagging][RF][Rotation Forests][Train] Raw data 
# =============================================================================
from sklearn.ensemble import BaggingClassifier # ensemble
from sklearn import tree

def Ensemble_clf(x_train, y_train, clf, n_clf, ens_title='', clf_name=''):   
    acc = []
    n_func = []
    
    for i in range(1, n_clf):        
        bagclf = BaggingClassifier(base_estimator=clf,n_estimators=i)
        bagclf = bagclf.fit(x_train, y_train)
        n_func.extend([i])
        acc.extend([1-bagclf.score(x_train, y_train)])
        
    acc_bag = pd.DataFrame({'Num_function':n_func, 'Accuracy':acc})
    
    # plot the error rate
    plt.figure()
    plt.plot(acc_bag.Num_function, acc_bag.Accuracy, '.-')
    plt.xticks(acc_bag.Num_function.values)
    plt.xlabel('Number of functions')
    plt.ylabel('Error rate')
    plt.title('Ensemble of classifiers {}'.format(clf_name))
    plt.suptitle('Dataset = {}'.format(ens_title))
    plt.show()
    return acc_bag

dt_clf = tree.DecisionTreeClassifier()
accuracy_tab = Ensemble_clf(x_train, y_train, dt_clf, 20, 'Bagging with Decision Tree')

# bagging with Decision Tree
bagclf = BaggingClassifier(base_estimator=dt_clf,n_estimators=10)
bagclf = bagclf.fit(x_train, y_train)


x_train.iloc[:,1:].head(10)


# =============================================================================
# [Bagging][RF][Rotation Forests][Train] SMOTE (under+over)
# =============================================================================

# create new dataset (balanced dataset)
from imblearn.combine import SMOTEENN

sm = SMOTEENN()
train_balanced, train_balanced_cl = sm.fit_sample(x_train, y_train)

# check the balanced dataset
# proportion of positive cases
original_prop = y_train.sum()[0]/x_train.shape[0]
balanced_prop = train_balanced_cl.sum()/train_balanced.shape[0]

print('<Proporsion of Positive Cases>')
print('Original   ->   Balanced')
print('%.4f     ->   %.4f' %(original_prop, balanced_prop))
print('\n')
print('<Number of Cases>')
print('Positive: {} -> {}'.format(y_train.sum()[0],train_balanced_cl.sum()))
print('Total:    {} -> {}'.format(y_train.shape[0],train_balanced.shape[0]))

















# Adaboost VS Rotation Forests


# Feature-Selection Ensembles (Rotation Forests)
















