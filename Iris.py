import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Cargar dataset
iris_df = pd.read_csv("iris.csv")
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Exploración de datos
print(iris_df.describe())
sns.pairplot(iris_df, hue='class', palette='Set1')
plt.suptitle("Relación entre características", y=1.02)
plt.show()

# Dividir datos
X = iris_df.drop('class', axis=1)
y = iris_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Modelos
models = {
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=3, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Regresión Logística": LogisticRegression(max_iter=200)
}

# Entrenamiento y evaluación
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"Matriz de Confusión - {name}")
    plt.show()
    
    results[name] = {
        "precision": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"],
        "recall": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"],
        "f1-score": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    }

# Comparación de métricas
df_results = pd.DataFrame(results).T
df_results.plot(kind='bar', figsize=(10, 6))
plt.title("Comparación de Métricas por Modelo")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()
