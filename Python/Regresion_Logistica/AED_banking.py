# Librerias
#-------------------------------------------------------------------------------
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


# Importación dataset
#-------------------------------------------------------------------------------
from pandas import read_csv
datos=read_csv('AED_banking.csv', header=0)
datos.head()


# Transformación
#-------------------------------------------------------------------------------
datos['education'].unique()


datos['education']=np.where(datos['education'] =='basic.9y', 'basic', datos['education'])
datos['education']=np.where(datos['education'] =='basic.6y', 'basic', datos['education'])
datos['education']=np.where(datos['education'] =='basic.4y', 'basic', datos['education'])

datos['education'].unique()


# Exploración
#-------------------------------------------------------------------------------
datos['y'].value_counts()

sns.countplot(x='y', data=datos, palette='hls')
plt.show()

count_no_sub = len(datos[datos['y']==0])
count_sub = len(datos[datos['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("El porcentaje de no suscripción de crétito a largo plazo es", pct_of_no_sub*100,"%")
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("El porcentaje de suscripción de crétito a largo plazo es", pct_of_sub*100,"%")

datos.groupby('y').mean()

datos.groupby('job').mean()

datos.groupby('marital').mean()

datos.groupby('education').mean()


# Visualizaciones
#-------------------------------------------------------------------------------
%matplotlib inline
pd.crosstab(datos.job,datos.y).plot(kind='bar')
plt.title('Frecuencia de compra en función de la profesión laboral') 
plt.xlabel('Cargo') 
plt.ylabel('Frecuencia de compra de crédito a largo plazo') 


table=pd.crosstab(datos.marital,datos.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title ('Gráfico de barras apiladas de educación frente a compras') 
plt.xlabel('Educación') 
plt.ylabel('Proporción de clientes') 


table=pd.crosstab(datos.education,datos.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Gráfico de barras apiladas de educación frente a compras') 
plt.xlabel('Educación') 
plt.ylabel('Proporción de clientes') 


pd.crosstab(datos.day_of_week,datos.y).plot(kind='bar')
plt.title('Frecuencia de compra por día de la semana') 
plt.xlabel('Día de la semana') 
plt.ylabel('Frecuencia de compra')


pd.crosstab(datos.month,datos.y).plot(kind='bar')
plt.title('Frecuencia de compra por mes') 
plt.xlabel('Mes') 
plt.ylabel('Frecuencia de compra') 


datos.age.hist()
plt.title('Histograma de edad') 
plt.xlabel('Edad') 
plt.ylabel('Frecuencia') 


pd.crosstab(datos.poutcome,datos.y).plot(kind='bar')
plt.title('Frecuencia de compra por Poutcome') 
plt.xlabel('Poutcome') 
plt.ylabel('Frecuencia de compra')


# Variables Ficticias
#-------------------------------------------------------------------------------
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(datos[var], prefix=var)
    datos1=datos.join(cat_list)
    datos=datos1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
datos_vars=datos.columns.values.tolist()
to_keep=[i for i in datos_vars if i not in cat_vars]

datos_final=datos[to_keep]
datos_final.columns.values


# Sobremuestreo usando SMOTE
#-------------------------------------------------------------------------------
X = datos_final.loc[:, datos_final.columns != 'y']
y = datos_final.loc[:, datos_final.columns == 'y']

from imblearn.over_sampling import SMOTE
oversample = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=oversample.fit_resample(X_train, y_train)

s_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# podemos comprobar los números de nuestros datos 
print("La longitud de los datos sobremuestreados es ",len(os_data_X)) 
print("Número de sin suscripción en datos sobremuestreados",len(os_data_y[os_data_y['y']==0])) 
print("Número de suscripción",len(os_data_y[os_data_y['y']==1])) 
print("La proporción de datos sin suscripción en datos sobremuestreados es ",len(os_data_y[os_data_y['y']==0])/ len(os_data_X)) 
print("La proporción de datos de suscripción en datos sobremuestreados es ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


datos_final_vars=datos_final.columns.values.tolist()
y=['y']
X=[i for i in datos_final_vars if i not in y]


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


cols=['marital_divorced', 'marital_married', 'marital_single', 'marital_unknown', 'education_basic', 'education_high.school', 'education_professional.course', 
      'education_university.degree', 'education_unknown', 'housing_no', 'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes', 
      'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu', "day_of_week_tue", "day_of_week_wed"] 
X=os_data_X[cols]
y=os_data_y['y']


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

