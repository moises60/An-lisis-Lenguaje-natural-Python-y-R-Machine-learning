import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("comida_a_domicilio_reviews.tsv", delimiter="\t", quoting=3)
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Configurar stopwords en español
stopwords_es = set(stopwords.words('spanish'))

# Inicializar el stemmer y lematizador
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(0, len(dataset)):
    # Eliminar caracteres especiales y números
    review = re.sub('[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', ' ', dataset['Review'][i])
    # Convertir a minúsculas
    review = review.lower()
    # Tokenización
    words = nltk.word_tokenize(review)
    # Eliminar stopwords y aplicar stemming/lemmatización
    words = [stemmer.stem(word) for word in words if word not in stopwords_es]
    # Unir las palabras procesadas
    review = ' '.join(words)
    corpus.append(review)

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked'].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

# Ajustar el clasificador Árbol de Decisión en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier, plot_tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy*100:.2f}%")

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar la matriz de confusión
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()


# Importancia de las características
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Mostrar las 20 características más importantes
top_features = 20
plt.figure(figsize=(10,5))
plt.title("Importancia de las características")
plt.bar(range(top_features), importances[indices][:top_features], align='center')
plt.xticks(range(top_features), [cv.get_feature_names_out()[i] for i in indices[:top_features]], rotation=90)
plt.tight_layout()
plt.show()
