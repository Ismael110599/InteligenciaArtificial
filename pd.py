import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================
# 1. Cargar y preparar los datos
# =======================
# Cargar el dataset
df = pd.read_csv("data.csv")

# Mostrar información inicial del dataset
print("Columnas disponibles:", df.columns.tolist())
print("\nPrimeras filas del dataset:")
print(df.head())

# =======================
# 2. Limpieza y selección de variables
# =======================
# Renombrar MSRP a price para mayor claridad
df = df.rename(columns={'MSRP': 'price'})

# Selección de variables relevantes (basado en las columnas disponibles)
relevant_columns = [
    'price', 
    'Year', 
    'Engine HP', 
    'Engine Cylinders', 
    'highway MPG', 
    'city mpg',
    'Number of Doors'
]

# Filtrar solo las columnas relevantes
df = df[relevant_columns]

# Renombrar columnas para mayor claridad
df = df.rename(columns={
    'Year': 'year',
    'Engine HP': 'engine_hp',
    'Engine Cylinders': 'engine_cylinders',
    'highway MPG': 'highway_mpg',
    'city mpg': 'city_mpg',
    'Number of Doors': 'doors'
})

# Eliminar filas con valores nulos
df = df.dropna()

# Verificar datos después de la limpieza
print("\nResumen después de la limpieza:")
print(df.info())

# =======================
# 3. Análisis exploratorio
# =======================
# Estadísticas generales
print("\nEstadísticas descriptivas:")
print(df.describe())

# Matriz de correlación
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de Correlación")
plt.show()

# Distribución del precio
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribución de Precios de Vehículos")
plt.xlabel("Precio (USD)")
plt.ylabel("Frecuencia")
plt.show()

# Relación entre año y precio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='price', data=df, alpha=0.6)
plt.title("Relación entre Año y Precio")
plt.xlabel("Año")
plt.ylabel("Precio (USD)")
plt.show()

# =======================
# 4. Preparación de datos para modelado
# =======================
# Variables predictoras y objetivo
X = df.drop('price', axis=1)
y = df['price']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTamaño de conjuntos:")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# =======================
# 5. Entrenamiento del modelo
# =======================
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficientes del modelo
print("\nCoeficientes del modelo:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercepto: {model.intercept_:.2f}")

# =======================
# 6. Evaluación del modelo
# =======================
# Predicciones
y_pred = model.predict(X_test)

# Métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo:")
print(f"Error Absoluto Medio (MAE): ${mae:,.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): ${rmse:,.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# =======================
# 7. Visualización de resultados
# =======================
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Precio Real (USD)")
plt.ylabel("Precio Predicho (USD)")
plt.title("Comparación: Precio Real vs. Precio Predicho")
plt.show()

# Residuos del modelo
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Precio Predicho (USD)")
plt.ylabel("Residuos")
plt.title("Análisis de Residuos")
plt.show()