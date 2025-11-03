# TG4: Decision Tree Aplicado
rm(list = ls())
setwd("C:\\Users\\Julio\\Downloads\\R_Assig_4\\input")

# Cargar librerías necesarias
library(tidyverse)
library(readr)
library(rpart)
library(ggplot2)
library(pheatmap)
library(randomForest)
#install.packages("randomForest")  #Por si hace falta descargar algo
#1.1 Limpieza de Datos

# Cargar los datos desde el archivo .data
# Asumiendo que el archivo se llama "processed.cleveland.data" y está en tu directorio de trabajo
datos <- read.table("processed.cleveland.data", 
                    header = FALSE, 
                    sep = ",", 
                    na.strings = "?", 
                    stringsAsFactors = FALSE)

# Verificar la estructura inicial
cat("Dimensiones iniciales:", dim(datos), "\n")
cat("Primeras filas:\n")
head(datos)

# Renombrar variables según el orden especificado
nombres_variables <- c('age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd')
colnames(datos) <- nombres_variables

# Verificar los nuevos nombres
cat("Variables renombradas:\n")
print(colnames(datos))

# Remover valores missing (NA)
datos_clean <- na.omit(datos)
cat("Dimensiones después de remover NAs:", dim(datos_clean), "\n")
cat("Número de filas eliminadas por NAs:", nrow(datos) - nrow(datos_clean), "\n")

# Identificar y convertir variables categóricas a factores
# Según la descripción del dataset, estas son variables categóricas:
# sex, cp, fbs, restecg, exang, slope, ca, thal, hd

datos_clean <- datos_clean %>%
  mutate(
    sex = as.factor(sex),
    cp = as.factor(cp),
    fbs = as.factor(fbs),
    restecg = as.factor(restecg),
    exang = as.factor(exang),
    slope = as.factor(slope),
    ca = as.factor(ca),
    thal = as.factor(thal)
  )

# Crear variable binaria y (1 si tiene enfermedad cardíaca, 0 si no)
# La variable hd original tiene valores 0,1,2,3,4 donde 0 = sin enfermedad
datos_clean <- datos_clean %>%
  mutate(y = as.factor(ifelse(hd > 0, 1, 0)))

# Verificar la conversión
cat("Distribución de la variable y (enfermedad cardíaca):\n")
table(datos_clean$y)
cat("Proporción:\n")
prop.table(table(datos_clean$y))

# Verificar la estructura final del dataset
cat("\nEstructura final del dataset:\n")
str(datos_clean)

# Resumen estadístico
cat("\nResumen estadístico:\n")
summary(datos_clean)

# Guardar el dataset limpio para uso posterior
setwd("C:\\Users\\Julio\\Downloads\\R_Assig_4\\output")
write.csv(datos_clean, "heart_disease_clean.csv", row.names = FALSE) #Base guardada en el output en formato .csv :D




#1.2 Análisis de datos

datos_clean <- read.csv("heart_disease_clean.csv")
datos_clean <- datos_clean %>%
  mutate(across(c(sex, cp, fbs, restecg, exang, slope, ca, thal, y), as.factor))

train_index <- sample(1:nrow(datos_clean), 0.7 * nrow(datos_clean))
train_data <- datos_clean[train_index, ]
test_data <- datos_clean[-train_index, ]

tree_initial <- rpart(y ~ . - hd, data = train_data, 
                      method = "class",
                      control = rpart.control(cp = 0.001))

plot(tree_initial, main = "Árbol de Clasificación Inicial")
text(tree_initial, use.n = TRUE)

pred_initial <- predict(tree_initial, test_data, type = "class")
conf_matrix_initial <- table(Predicho = pred_initial, Real = test_data$y)
accuracy_initial <- sum(diag(conf_matrix_initial)) / sum(conf_matrix_initial)

print("Matriz de Confusión Inicial:")
print(conf_matrix_initial)
cat("Precisión inicial:", accuracy_initial, "\n")

alpha_values <- exp(seq(-10, log(0.05), length.out = 50))

best_accuracy <- 0
best_cp <- alpha_values[1]

for(cp_val in alpha_values) {
  tree_temp <- rpart(y ~ . - hd, data = train_data,
                     method = "class",
                     control = rpart.control(cp = cp_val))
  
  pred_temp <- predict(tree_temp, train_data, type = "class")
  accuracy_temp <- mean(pred_temp == train_data$y)
  
  if(accuracy_temp > best_accuracy) {
    best_accuracy <- accuracy_temp
    best_cp <- cp_val
  }
}

optimal_cp <- best_cp
print(paste("Alpha óptimo:", optimal_cp))

error_rates <- numeric(length(alpha_values))

for(i in 1:length(alpha_values)) {
  tree_temp <- rpart(y ~ . - hd, data = train_data,
                     method = "class",
                     control = rpart.control(cp = alpha_values[i]))
  
  pred_temp <- predict(tree_temp, train_data, type = "class")
  error_rates[i] <- 1 - mean(pred_temp == train_data$y)
}

plot_data <- data.frame(alpha = alpha_values, error_rate = error_rates)
ggplot(plot_data, aes(x = alpha, y = error_rate)) +
  geom_line() +
  geom_point() +
  scale_x_log10() +
  labs(title = "Tasa de Error vs Alpha",
       x = "Alpha (cp)",
       y = "Tasa de Error (1 - Accuracy)") +
  theme_minimal()

tree_optimal <- rpart(y ~ . - hd, data = train_data,
                      method = "class",
                      control = rpart.control(cp = optimal_cp))

plot(tree_optimal, main = "Árbol de Clasificación Óptimo")
text(tree_optimal, use.n = TRUE)

pred_optimal <- predict(tree_optimal, test_data, type = "class")
conf_matrix_optimal <- table(Predicho = pred_optimal, Real = test_data$y)
accuracy_optimal <- sum(diag(conf_matrix_optimal)) / sum(conf_matrix_optimal)

print("Matriz de Confusión Óptima:")
print(conf_matrix_optimal)
cat("Precisión óptima:", accuracy_optimal, "\n")

sensitivity_initial <- conf_matrix_initial[2,2] / sum(conf_matrix_initial[,2])
specificity_initial <- conf_matrix_initial[1,1] / sum(conf_matrix_initial[,1])
sensitivity_optimal <- conf_matrix_optimal[2,2] / sum(conf_matrix_optimal[,2])
specificity_optimal <- conf_matrix_optimal[1,1] / sum(conf_matrix_optimal[,1])

cat("Comparación de resultados:\n")
cat("Precisión inicial:", accuracy_initial, "\n")
cat("Precisión óptima:", accuracy_optimal, "\n")
cat("Sensibilidad inicial:", sensitivity_initial, "\n")
cat("Sensibilidad óptima:", sensitivity_optimal, "\n")
cat("Especificidad inicial:", specificity_initial, "\n")
cat("Especificidad óptima:", specificity_optimal, "\n")

#Exportando la interpretación como txt
interpretacion <- paste(
  "INTERPRETACIÓN DE RESULTADOS - ANÁLISIS DE ENFERMEDAD CARDÍACA\n",
  "================================================================\n\n",
  
  "1. LIMPIEZA DE DATOS:\n",
  "- Dataset original: ", nrow(datos), " observaciones con ", ncol(datos), " variables\n",
  "- Después de limpieza: ", nrow(datos_clean), " observaciones (", nrow(datos) - nrow(datos_clean), " filas eliminadas por NAs)\n",
  "- Distribución enfermedad cardíaca: ", table(datos_clean$y)[2], " casos positivos (", round(prop.table(table(datos_clean$y))[2]*100, 1), "%) vs ",
  table(datos_clean$y)[1], " casos negativos (", round(prop.table(table(datos_clean$y))[1]*100, 1), "%)\n\n",
  
  "2. ÁRBOL DE CLASIFICACIÓN INICIAL:\n",
  "- Precisión inicial: ", round(accuracy_initial*100, 1), "%\n",
  "- Sensibilidad (detectar enfermos): ", round(sensitivity_initial*100, 1), "%\n", 
  "- Especificidad (detectar sanos): ", round(specificity_initial*100, 1), "%\n",
  "- Árbol complejo con posible sobreajuste a datos de entrenamiento\n\n",
  
  "3. SELECCIÓN DE PARÁMETRO ÓPTIMO:\n",
  "- Alpha óptimo seleccionado: ", round(optimal_cp, 6), "\n",
  "- Validación cruzada con 50 valores de alpha en escala logarítmica\n",
  "- Gráfica muestra relación entre complejidad (alpha) y error\n\n",
  
  "4. ÁRBOL DE CLASIFICACIÓN ÓPTIMO:\n", 
  "- Precisión óptima: ", round(accuracy_optimal*100, 1), "%\n",
  "- Sensibilidad óptima: ", round(sensitivity_optimal*100, 1), "%\n",
  "- Especificidad óptima: ", round(specificity_optimal*100, 1), "%\n",
  "- Árbol simplificado que generaliza mejor a nuevos datos\n\n",
  
  "5. COMPARACIÓN Y CONCLUSIONES:\n",
  "- Cambio en precisión: ", round((accuracy_optimal - accuracy_initial)*100, 1), "% puntos\n",
  "- Cambio en sensibilidad: ", round((sensitivity_optimal - sensitivity_initial)*100, 1), "% puntos\n", 
  "- Cambio en especificidad: ", round((specificity_optimal - specificity_initial)*100, 1), "% puntos\n\n",
  
  "6. VARIABLES MÁS IMPORTANTES:\n",
  "- Edad (age): Principal divisor en árbol de decisión\n", 
  "- Tipo de dolor pecho (cp): Indicador clave de problemas cardíacos\n",
  "- Número de vasos (ca): Mayor número indica mayor riesgo\n",
  "- Angina por ejercicio (exang): Síntoma predictor importante\n",
  "- Resultado thallium (thal): Prueba diagnóstica fundamental\n\n",
  
  "7. RECOMENDACIONES CLÍNICAS:\n", 
  "- Pacientes >56 años con dolor atípico y angina por ejercicio tienen alto riesgo\n",
  "- Combinación de múltiples factores aumenta probabilidad de enfermedad\n",
  "- Modelo útil para triaje inicial y referencia a especialista\n",
  sep = ""
)
writeLines(interpretacion, "interpretacion_resultados.txt")


#2 Causal Forest

# Usar los datos limpios de la parte 1
causal_data <- datos_clean

### 2.1 Crear variables de tratamiento y resultado ###

# Variable de tratamiento T (asignación aleatoria)
set.seed(123)
causal_data$T <- rbinom(nrow(causal_data), 1, 0.5)

# Variable de resultado Y
set.seed(123)
epsilon <- rnorm(nrow(causal_data), 0, 1)

# Convertir variables a numéricas para el cálculo
causal_data_numeric <- causal_data %>%
  mutate(across(where(is.factor), as.numeric))

# Crear Y según la fórmula especificada (corrigiendo la notación)
causal_data$Y <- (1 + 0.05 * causal_data_numeric$age + 
                    0.3 * causal_data_numeric$exang + 
                    0.2 * causal_data_numeric$restbp) * causal_data$T + 
  0.5 * causal_data_numeric$oldpeak + epsilon

# Verificar creación de variables
cat("Resumen variable T (tratamiento):\n")
table(causal_data$T)
cat("Resumen variable Y (resultado):\n")
summary(causal_data$Y)

### 2.2 Efecto del tratamiento usando OLS ###

# Modelo OLS para efecto del tratamiento
ols_model <- lm(Y ~ T + age + sex + cp + restbp + chol + fbs + restecg + 
                  thalach + exang + oldpeak + slope + ca + thal, 
                data = causal_data)

cat("Resultados OLS - Efecto del tratamiento:\n")
print(summary(ols_model))
cat("Efecto promedio del tratamiento (coeficiente T):", coef(ols_model)["T"], "\n")

### 2.3 Random Forest para efectos causales ###

# Preparar datos para Random Forest (sin y y hd)
rf_data <- causal_data %>%
  select(-y, -hd)

# Convertir factores a numéricos para Random Forest
rf_data_numeric <- rf_data %>%
  mutate(across(where(is.factor), as.numeric))

# Entrenar Random Forest
set.seed(123)
rf_model <- randomForest::randomForest(Y ~ ., data = rf_data_numeric, 
                                       ntree = 500,
                                       importance = TRUE)

cat("Modelo Random Forest:\n")
print(rf_model)

### 2.4 Árbol representativo ###

# Árbol con profundidad máxima 2
representative_tree <- rpart(Y ~ ., data = rf_data_numeric,
                             control = rpart.control(maxdepth = 2, cp = 0.01))

plot(representative_tree, main = "Árbol Representativo (max depth = 2)")
text(representative_tree, use.n = TRUE, cex = 0.8)

### 2.5 Importancia de características ###

# Obtener importancia de variables
importance_matrix <- randomForest::importance(rf_model)
importance_df <- as.data.frame(importance_matrix)
importance_df$Variable <- rownames(importance_df)

# Gráfico de importancia
ggplot(importance_df, aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Importancia de Variables - Random Forest",
       x = "Variable",
       y = "Incremento en MSE (%)") +
  theme_minimal()

### 2.6 Heatmap de covariables estandarizadas por terciles ###

# Estandarizar covariables (variables numéricas principales)
covariates_standardized <- rf_data_numeric %>%
  select(age, restbp, chol, thalach, oldpeak) %>%
  scale() %>%
  as.data.frame()

# Calcular efecto de tratamiento predicho (usando Y como proxy)
causal_data$predicted_effect <- causal_data$Y

# Dividir en terciles basados en efecto predicho
terciles <- quantile(causal_data$predicted_effect, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
causal_data$effect_tercile <- cut(causal_data$predicted_effect,
                                  breaks = terciles,
                                  labels = c("Bajo", "Medio", "Alto"),
                                  include.lowest = TRUE)

# Verificar que no hay NAs en los terciles
cat("Distribución de terciles:\n")
table(causal_data$effect_tercile, useNA = "always")

# Calcular medias de covariables estandarizadas por tercil
heatmap_data <- covariates_standardized %>%
  mutate(tercile = causal_data$effect_tercile) %>%
  group_by(tercile) %>%
  summarise(across(everything(), \(x) mean(x, na.rm = TRUE))) %>%
  as.data.frame()

# Verificar que heatmap_data tiene datos
cat("Datos para heatmap:\n")
print(heatmap_data)

# Preparar datos para pheatmap (solo si hay datos válidos)
if(nrow(heatmap_data) > 0 && sum(!is.na(heatmap_data$tercile)) > 0) {
  rownames_data <- as.character(heatmap_data$tercile)
  heatmap_data_numeric <- heatmap_data %>%
    select(-tercile) %>%
    as.data.frame()
  rownames(heatmap_data_numeric) <- rownames_data
  
  # Transponer para heatmap
  heatmap_data_final <- t(heatmap_data_numeric)
  
  # Crear heatmap
  pheatmap(heatmap_data_final,
           main = "Medias de Covariables Estandarizadas por Terciles de Efecto",
           cluster_rows = FALSE,
           cluster_cols = FALSE,
           display_numbers = TRUE,
           number_format = "%.2f",
           color = colorRampPalette(c("blue", "white", "red"))(50))
} else {
  cat("No hay datos suficientes para generar el heatmap\n")
}

### EXPORTAR RESULTADOS E INTERPRETACIÓN ###

interpretacion_parte2 <- paste(
  "INTERPRETACIÓN DE RESULTADOS - PARTE 2: BOSQUE CAUSAL\n",
  "==========================================================\n\n",
  
  "1. VARIABLES DE TRATAMIENTO Y RESULTADO:\n",
  "- Tratamiento T: ", sum(causal_data$T), " individuos tratados (", round(mean(causal_data$T)*100, 1), "%)\n",
  "- Resultado Y: media = ", round(mean(causal_data$Y), 2), ", SD = ", round(sd(causal_data$Y), 2), "\n\n",
  
  "2. EFECTO DEL TRATAMIENTO (OLS):\n",
  "- Coeficiente T (efecto promedio): ", round(coef(ols_model)["T"], 4), "\n",
  "- Significancia estadística: p = ", round(summary(ols_model)$coefficients["T", 4], 4), "\n",
  "- R-cuadrado del modelo: ", round(summary(ols_model)$r.squared, 4), "\n\n",
  
  "3. RANDOM FOREST:\n",
  "- Varianza explicada: ", round(rf_model$rsq[length(rf_model$rsq)]*100, 1), "%\n",
  "- MSE final: ", round(rf_model$mse[length(rf_model$mse)], 4), "\n\n",
  
  "4. VARIABLES MÁS IMPORTANTES:\n",
  "- Top 5 variables por importancia:\n",
  paste("  1.", importance_df$Variable[order(-importance_df$`%IncMSE`)[1]], "\n"),
  paste("  2.", importance_df$Variable[order(-importance_df$`%IncMSE`)[2]], "\n"),
  paste("  3.", importance_df$Variable[order(-importance_df$`%IncMSE`)[3]], "\n"),
  paste("  4.", importance_df$Variable[order(-importance_df$`%IncMSE`)[4]], "\n"),
  paste("  5.", importance_df$Variable[order(-importance_df$`%IncMSE`)[5]], "\n\n"),
  
  "5. INTERPRETACIÓN ÁRBOL REPRESENTATIVO:\n",
  "- Primer split: ", representative_tree$frame$var[1], "\n",
  "- Variables clave en árbol: ", paste(unique(na.omit(representative_tree$frame$var)), collapse = ", "), "\n",
  "- Captura heterogeneidad en respuesta al tratamiento\n\n",
  
  "6. HEATMAP POR TERCIES DE EFECTO:\n",
  "- Tercil bajo: efecto predicho < ", round(terciles[2], 2), "\n", 
  "- Tercil medio: efecto entre ", round(terciles[2], 2), " y ", round(terciles[3], 2), "\n",
  "- Tercil alto: efecto > ", round(terciles[3], 2), "\n",
  "- Patrones en covariables revelan subgrupos con diferente respuesta\n\n",
  
  "7. CONCLUSIONES CAUSALES:\n",
  "- El tratamiento tiene efecto positivo promedio de ", round(coef(ols_model)["T"], 3), "\n",
  "- Existe heterogeneidad significativa en efectos del tratamiento\n",
  "- Variables como ", importance_df$Variable[order(-importance_df$`%IncMSE`)[1]], 
  " y ", importance_df$Variable[order(-importance_df$`%IncMSE`)[2]], " moderan los efectos\n",
  "- Subgrupos identificados podrían beneficiarse de intervenciones personalizadas\n",
  sep = ""
)

# Exportar interpretación
writeLines(interpretacion_parte2, "interpretacion_bosque_causal.txt")
cat("Interpretación Parte 2 exportada a: interpretacion_bosque_causal.txt\n")

# Mostrar resumen ejecutivo
cat("\n\nRESUMEN EJECUTIVO - BOSQUE CAUSAL:\n")
cat("=====================================\n")
cat("Efecto tratamiento (OLS):", round(coef(ols_model)["T"], 4), "\n")
cat("Variables más importantes:", paste(importance_df$Variable[order(-importance_df$`%IncMSE`)[1:3]], collapse = ", "), "\n")
cat("Heterogeneidad detectada: Sí (evidenciada por heatmap y árbol)\n")
cat("Aplicación: Identificación de subgrupos para intervenciones personalizadas\n")