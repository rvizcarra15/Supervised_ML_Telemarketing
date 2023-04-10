# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:21:51 2023

@author: saga
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#Importar datos CSV desde ruta*#
#import pandas as pd
#df = pd.read_csv(r'D:\RVC\Academico\PYTHON\Ejercicios\MKT_BANK\prueba_carga.csv')
#print(df)

#Importando data de Banco#
df = pd.read_csv(r'D:\RVC\Academico\PYTHON\Ejercicios\MKT_BANK\banking.txt')

# Return the first 5 rows of the DataFrame.
print(df.head())
print()
# Ver Contenido de un campo
print(df['education'].unique())
print()
# Reagrupamos un campo***como un REPLACE****
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
print(df['education'].unique())
print()

#Exploracion de los datos
#valores 1 y 0
print(df['y'].value_counts())
#enviados a una tabla
y_counts_df = pd.DataFrame(df['y'].value_counts())
# grafico en pie
# se neceesitan estas librerias
#import matplotlib.pyplot as plt
#import numpy as np
# Defining colors for the pie chart
import pandas as pd
colors = ['steelblue', 'gold']
labels = ['No Compraron', 'Compraron']
df.groupby(['y']).count().plot(kind='pie',y = 'age',autopct='%1.0f%%',shadow=True,
                               colors=colors,labels=labels) 
plt.title("Distribución de Muestra por resultado", bbox={'facecolor':'0.8', 'pad':5})
plt.show()


#HISTOGRAMA
df.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#CORRELACION
Corr_matrix= df.corr()
round(Corr_matrix,2)
sns.heatmap(Corr_matrix)
plt.show()
#grafico distribucion porcentual
table=pd.crosstab(df.marital,df.y)
cross_tab_prop = pd.crosstab(index=df['marital'],
                             columns=df['y'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6),color=['steelblue', 'gold'])

plt.legend(['No Compro','Compro'],loc="lower center", ncol=2)
plt.xlabel('Marital Status')
plt.ylabel("Proportion")
plt.title('Stacked Bar Chart of Marital Status vs Purchase', bbox={'facecolor':'0.8', 'pad':5})

for n, x in enumerate([*table.index.values]):
    for (proportion, count, y_loc) in zip(cross_tab_prop.loc[x],
                                          table.loc[x],
                                          cross_tab_prop.loc[x].cumsum()):
                
        plt.text(x=n - 0.17,
                 y=(y_loc - proportion) + (proportion / 2),
                 s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                 color="black",
                 fontsize=12,
                 fontweight="bold")
plt.show()
#BARRAS HORIZONTALES

table_horiz= pd.crosstab(df.education,df.y)
cross_tab_horiz= pd.crosstab(index=df['education'],
                             columns=df['y'],
                             normalize="index")
cross_tab_horiz.plot(kind='barh', 
                        stacked=True, 
                        colormap='tab10', 
                        figsize=(10, 6))

plt.legend(loc="lower left", ncol=2)
plt.ylabel("Education")
plt.xlabel("Proportion")
plt.title('Stacked Bar Chart of Education vs Purchase', bbox={'facecolor':'0.8', 'pad':5})

for n, x in enumerate([*table_horiz.index.values]):
    for (proportion, count, y_loc) in zip(cross_tab_horiz.loc[x],
                                          table_horiz.loc[x],
                                          cross_tab_horiz.loc[x].cumsum()):
                
        plt.text(x=(y_loc - proportion) + (proportion / 2),
                 y=n - 0.11,
                 s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                 color="black",
                 fontsize=12,
                 fontweight="bold")
plt.show()
y_media = pd.DataFrame(df.groupby('y').mean())
#***********************TRANSFORMACION A DUMMIES********************************#
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=df[to_keep]
data_final.columns.values
#CORRELACION
#Corr_matrix_final= data_final.corr()
#round(Corr_matrix_final,2)
#sns.heatmap(Corr_matrix_final)

#************************( muestra balanceada-Metodo SMOTE)**************************************
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#***************************Recursive Feature Elimination (RFE)****************************************************
#Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model 
#and choose either the best or worst performing feature, setting the feature aside and then repeating
#the process with the rest of the features. This process is applied until all features in the dataset are exhausted. 

data_final_vars = data_final.columns.values.tolist()
y = ['y']
X = [i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(estimator=logreg, n_features_to_select=20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
Rkg_columns = pd.DataFrame(rfe.ranking_)
print(Rkg_columns)

#***************************MATCH MEJORES VARIABLES******
Rkg_columns.to_csv('BEST_RKG_COLUMNS.csv')
os_data_X.to_csv('PRE_DATA_LOGIT.csv')

#*********INCORPORAMOS LAS MEJORES VARIABLES SELECCIONADAS EN RFE********************

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

#*******************************IMPLEMENTAMOS MODELO*******************************
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#**********************************REMOVIENDO VARIABLES CON P-VALUE>5***************
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(result.summary2()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')
plt.show()

#**********************Logistic Regression Model Fitting*****************************

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)







#*********************CONFUSION MATRIX***************************************************

y_pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(confusion_matrix)
print()

# Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy')
print(accuracy_score(y_test, y_pred))
print()
# Precision
print('Precision')
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred, average=None))
print()
# Recall
from sklearn.metrics import recall_score
print('Recall')
print(recall_score(y_test, y_pred, average=None))
print()
# F1-SCORE
print('F1-SCORE')
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average=None))
print()
# AUC SCORE
print('AUC SCORE')
from sklearn import metrics
#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred)
print(auc)

print('Confusion Matrix_Metrics')
# Precision y Recall are the line in predicted (1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Print the confusion matrix using Seaborn
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                confusion_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     confusion_matrix.flatten()/np.sum(confusion_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()


#*********************ROC Curve**************************************************
#The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers.
#The dotted line represents the ROC curve of a purely random classifier; 
#a good classifier stays as far away from that line as possible (toward the top-left corner).
#The closer AUC is to 1, the better the model. 
#A model with an AUC equal to 0.5 is no better than a model 
#that makes random classifications.


#ROC curve, which is a plot that displays 
#the sensitivity and specificity of a logistic regression model.




from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate-(1-Specificity)')
plt.ylabel('True Positive Rate-Sensitivity/Recall')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#*********************Precision-Recall Curve***********************************
#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

#*********regression model VS gradient boosted model classifier ***************
#

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
#model = LogisticRegression()
#model.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
#https://vagifaliyev.medium.com/a-hands-on-explanation-of-gradient-boosting-regression-4cfe7cfdf9e

#While Gradient Boosting is an Ensemble Learning method,
# it is more specifically a Boosting Technique
#In Gradient Boosting, each predictor tries to improve on its predecessor 
#by reducing the errors. 
#The two most popular boosting methods are:
#1)Adaptive Boosting
#2)Gradient Boosting

#The learning_rate is a hyperparameter that is used to scale 
#each trees contribution, sacrificing bias for better variance

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred_GB = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_GB)
auc = round(metrics.roc_auc_score(y_test, y_pred_GB), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

#*****KNN
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred_KNN = classifier.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_KNN)
auc = round(metrics.roc_auc_score(y_test, y_pred_KNN), 4)
plt.plot(fpr,tpr,label="KNN Classifier, AUC="+str(auc))

#add legend
plt.legend()


#model = LogisticRegression()
#model.fit(X_train, y_train)
#y_pred = model.predict_proba(X_test)[:, 1]
#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
#auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
#plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))
















