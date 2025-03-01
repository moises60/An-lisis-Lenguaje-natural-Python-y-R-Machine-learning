# Cargar librer�as necesarias
library(tm)
library(SnowballC)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(ggplot2)

# Importar el dataset con encoding UTF-8
dataset <- read.delim("comida_a_domicilio_reviews.tsv", header = TRUE, stringsAsFactors = FALSE, fileEncoding = "UTF-8")

# Verificar los nombres de las columnas
colnames(dataset)
# Deber�as ver 'Review' y 'Liked'

# Asegurar que 'Liked' es un factor
dataset$Liked <- as.factor(dataset$Liked)

# Inspeccionar el dataset
str(dataset)
head(dataset)

# Preprocesamiento de texto
corpus <- VCorpus(VectorSource(dataset$Review))

clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("[^a-z�������]", " ", text)
  text <- stripWhitespace(text)
  text <- removeWords(text, stopwords("spanish"))
  text <- wordStem(text, language = "spanish")
  return(text)
}

corpus_clean <- tm_map(corpus, content_transformer(clean_text))

# Crear la matriz de t�rminos (Bag of Words)
dtm <- DocumentTermMatrix(corpus_clean)
dtm <- removeSparseTerms(dtm, 0.995)

# Convertir la matriz a un dataframe
X <- as.data.frame(as.matrix(dtm))
data <- cbind(X, Liked = dataset$Liked)

# Dividir el dataset en conjunto de entrenamiento y prueba
set.seed(123)
trainIndex <- createDataPartition(data$Liked, p = 0.65, list = FALSE)
data_train <- data[trainIndex, ]
data_test <- data[-trainIndex, ]

# Entrenar el clasificador �rbol de Decisi�n
model <- rpart(Liked ~ ., data = data_train, method = "class")

# Predecir en el conjunto de prueba
predictions <- predict(model, newdata = data_test, type = "class")

# Evaluar el modelo
conf_matrix <- confusionMatrix(predictions, data_test$Liked)
print(conf_matrix)

# Visualizar la matriz de confusi�n
conf_table <- as.data.frame(conf_matrix$table)
colnames(conf_table) <- c("Prediction", "Reference", "Freq")

ggplot(data = conf_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Matriz de Confusi�n", x = "Valor Real", y = "Predicci�n") +
  theme_minimal()

# Visualizar el �rbol de decisi�n
rpart.plot(model, type = 2, extra = 104, fallen.leaves = TRUE, main = "�rbol de Decisi�n")

# Importancia de las variables
importance <- varImp(model, scale = FALSE)
importance_df <- data.frame(Variables = rownames(importance), Importance = importance$Overall)
importance_df <- importance_df[order(-importance_df$Importance), ]

# Mostrar las 20 variables m�s importantes
top_variables <- head(importance_df, 20)
ggplot(top_variables, aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Importancia de las Variables", x = "Variables", y = "Importancia") +
  theme_minimal()


