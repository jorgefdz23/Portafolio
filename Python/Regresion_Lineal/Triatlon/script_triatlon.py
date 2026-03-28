# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Importación de dataset
df_triatlon = pd.read_excel('triathlon.xlsx')
print(df_triatlon.shape)
df_triatlon.head()

# ==============================================================================
# Correlación
corr_matrix = df_triatlon.select_dtypes(include=['float64', 'int']).corr(method='pearson')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 8},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)

sns.pairplot(df_triatlon)

# ==============================================================================
# Pruebas de Regresión

# Creación de train y test
train = df_triatlon[df_triatlon['Sample_ID']==1]
X_train = train[['Edad', 'Peso', 'Experiencia', 'EnCarrera', 'EnBici',
                       'EnNatación', 'CoCarrera', 'CoBici', 'CoNatación']]
y_train = train[['Tiempo']]

test = df_triatlon[df_triatlon['Sample_ID']!=1]
X_test = test[['Edad', 'Peso', 'Experiencia', 'EnCarrera', 'EnBici',
                       'EnNatación', 'CoCarrera', 'CoBici', 'CoNatación']]
y_test = test[['Tiempo']]


# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================

datos_train = pd.DataFrame(
                    np.hstack((X_train, y_train)),
                    columns=['Edad', 'Peso', 'Experiencia', 'EnCarrera', 'EnBici',
                           'EnNatación', 'CoCarrera', 'CoBici', 'CoNatación', 'Tiempo'] )

modelo = smf.ols(formula = 'Tiempo ~ Edad + Peso + Experiencia + EnCarrera + EnBici + \
                            EnNatación + CoCarrera + CoBici + CoNatación ', data = datos_train)
modelo = modelo.fit()
print(modelo.summary())


# Ajustes del modelo sin variable significativas
# ==============================================================================
modelo = smf.ols(formula = 'Tiempo ~ Edad  + Experiencia + EnCarrera + EnBici + \
                            CoCarrera', data = datos_train)
modelo = modelo.fit()
print(modelo.summary())

X_train_2 = train[['Edad', 'Experiencia', 'EnCarrera', 'EnBici', 'CoCarrera']]
X_test_2 = test[['Edad', 'Experiencia', 'EnCarrera', 'EnBici', 'CoCarrera']]

y_train = y_train.to_numpy().flatten()
prediccion_train = modelo.predict(exog = X_train_2)
residuos_train   = prediccion_train - y_train


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.5)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'k--', color = 'black', lw=2)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('DataReal', fontsize = 7,fontweight = "bold")
axes[0, 0].set_ylabel('DataPredicción', fontsize = 7,fontweight = "bold")
axes[0, 0].tick_params(labelsize = 7)



axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id', fontsize = 7,fontweight = "bold")
axes[0, 1].set_ylabel('Residuo', fontsize = 7,fontweight = "bold")
axes[0, 1].tick_params(labelsize = 7)


sns.histplot(
    data    = residuos_train,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)

axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    residuos_train,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

axes[2, 0].scatter(prediccion_train, residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
axes[2, 0].set_xlabel('Predicción')
axes[2, 0].set_ylabel('Residuo')
axes[2, 0].tick_params(labelsize = 7)

# Se eliminan los axes vacíos
fig.delaxes(axes[2,1])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");


# Normalidad de los residuos Shapiro-Wilk test
# ==============================================================================
shapiro_test = stats.shapiro(residuos_train)
shapiro_test


# Normalidad de los residuos D'Agostino's K-squared test
# ==============================================================================
k2, p_value = stats.normaltest(residuos_train)
print(f"Estadítico= {k2}, p-value = {p_value}")


# Error de test del modelo 
# ==============================================================================
X_test_2 = sm.add_constant(X_test_2, prepend=True)
predicciones = modelo.predict(exog = X_test_2)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")


# Error de train del modelo 
# ==============================================================================
X_train_2 = sm.add_constant(X_train_2, prepend=True)
predicciones = modelo.predict(exog = X_train_2)
rmse = mean_squared_error(
        y_true  = y_train,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de tran es: {rmse}")

