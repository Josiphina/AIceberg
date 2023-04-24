import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc , confusion_matrix
import matplotlib.pyplot as plt
import itertools

plt.style.use('ggplot')

path =r'/results_unbiased.csv'
dataset2 = pd.read_csv(path)
seed = 50


# #feature selection
# cols = [4, 5, 6, 7, 8, 9, 10]
# ans2= list(itertools.combinations(cols, 2))
# ans3 = list(itertools.combinations(cols, 3))
# ans4 = list(itertools.combinations(cols, 4))
# ans5 = list(itertools.combinations(cols, 5))
# ans6 = list(itertools.combinations(cols, 6))
# asn7 = list(itertools.combinations(cols, 7))
# answer = [*ans2 , *ans3 , *ans4 , *ans5, *ans6, *asn7]



#load models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('SVM', SVC(probability=True)))


false_positive_rate = [0]*5
true_positive_rate = [0]*5
threshold = [0]*5
#i=0
scoring = 'accuracy'

fig = plt.figure(figsize=(10,10))
plt.rcParams.update(plt.rcParamsDefault)   
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30}) 

# k=0
# featuresel = pd.DataFrame(columns=['selection', 'name', 'rocscore'])
# for item in answer:
col = [6, 7, 8]
X = dataset2.iloc[:19379 , col]; y = dataset2.iloc[:19379 , 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=seed)
results = []
y_score = []
i=0
#iterate over all models
for name, model in models:
    model.fit(X_train , y_train)
    if name == 'SVM':
        y_prob = model.decision_function(X_test)
        prob_pos = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        y_score.append(prob_pos)
    else:
        y_score.append(model.predict_proba(X_test)[:, 1] )
    false_positive_rate[i] , true_positive_rate[i] , threshold[i] = roc_curve(y_test , y_score[i])
    score = roc_auc_score(y_test, y_score[i])
    print('roc_auc_score for %s:' % (name), score )
    #featuresel.loc[k] =[col, name , score] 
    #k = k+1

# featuresel.to_csv('featuresel.csv', sep=';')
# maxind = featuresel[['rocscore']].idxmax()
# print( maxind , featuresel.iloc[maxind, :], featuresel[['rocscore']].max() )

    
    #plot ROC curves
    lines = ['--', '-', '-.', ':']
    plt.plot(false_positive_rate[i], true_positive_rate[i] , linestyle = lines[i] , linewidth =5 , label ='%s' % (name))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate', labelpad=20)
    plt.xlabel('False Positive Rate', labelpad=20)
    i= i+1


 
plt.legend()
plt.savefig('ROC_curves.pdf')
plt.show()

