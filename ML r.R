
###### MODELOS DE PRONÓSTICOS PARA PREDECIR SERIES CON MACHINE LEARNING

###### VAMOS A PREDECIR LA INFLACIÓN

library(ggplot2)
library(caret)
library(readxl)
library(dplyr)
library("dplyr")
library("faux")
library("DataExplorer")
library("randomForest")

##Importar una base de datos con las variables de inflación y un gran conjunto
## de variables adicionales para usarlas como predictores a la hora de pronosticar.

data <- read_excel("D:/Usuario/Desktop/data inflacion.xls")
data_com <- data


y_var_completo <- data[,1]
# Quitamos octubre de la base
data <- data[1:(nrow(data)-1),]


# ELIMINACIÓN RECURSIVA MEDIANTE RANDOM FOREST Y BOOTSTRAPPING

#Primero, de un conjunto grande de variables, utilizaremos métodos para definir el
#subconjunto de variables que mejor explican el comportamiento de la serie. 

subsets <- c(3:11)
set.seed(123)


ctrl_rfe <- rfeControl(functions = rfFuncs, method = "boot", number = repeticiones,
                       returnResamp = "all", verbose = FALSE,
                       seeds = seeds)



control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 5, # number of repeats
                      number = 10) # number of folds


# Se ejecuta la eliminación recursiva de predictores
set.seed(342)

rf_rfe <- rfe(inf_sub~., data = data,
              sizes = subsets,
              rfeControl = control)



rf_rfe 

predictors(rf_rfe)


ggplot(data = rf_rfe) + theme_bw() + geom_line(lwd=0.9) + geom_point(lwd=1.1)


### Teniendo en cuenta lo anterior nos quedamos con 11 variables 
#inf_sub inf_sub_l1 tasa_depo_6_l base_mon_l tasa_depo_3_var_anu_l 
#ipp_l m2_l2 m1_l2 tasa_depo_6_var_anu_l per_ocu_manu_l2  tasa_depo_3_l 
# cre_int_var_anual_l



####### Proceso de predicción

data <- read_excel("D:/Usuario/Desktop/Tercer Semestre/Econometria Aplicada/Tarea 3/Problema 3/Mensuales para Inflación/data_inflacion_final.xls")
data_com <- data

y_var_completo <- data[,1]
# Quitamos octubre de la base
data <- data[1:(nrow(data)-1),]

# Usar 10-fold cross-validation con k=10 para todos los métodos que siguen
ctrl <- trainControl(method="cv", number=10)

######## MODELOS BASICOS

###No olvidar que el modelo se entrena con una fracción de la información estadística de las series. 

#Regresión lineal
modelo1 <- train(inf_sub~., data=data, method="lm", trControl=ctrl)
modelo1

# KNN sin preprocesar
modelo2a <- train(inf_sub~., data=data, method="knn", trControl=ctrl)
modelo2a

# KNN con preprocesamiento
modelo2b <- train(inf_sub~., data=data, method="knn", preProcess=c("center", "scale"), trControl=ctrl)
modelo2b

# KNN con preprocesamiento y grid search para k
knnGrid <- expand.grid(k=c(1,5,10,30,100))
modelo2c <- train(inf_sub~., data=data, method="knn", preProcess=c("center", "scale"), tuneGrid=knnGrid, trControl=ctrl)
modelo2c

# CART: Parametro de control = maxdepth (profundidad maxima del arbol)
modelo3a <- train(inf_sub~., data=data, method="rpart2", trControl=ctrl)
modelo3a

# CART: busqueda de mejor maxdepth
cartGrid <- expand.grid(maxdepth=c(1,5,10,20))
modelo3b <- train(inf_sub~., data=data, method="rpart2", tuneGrid=cartGrid, trControl=ctrl)
modelo3b


##### AVANZADOS

# Random Forest

modelo4 <- train(inf_sub~., data=data, method="rf", tunelength=6, trControl=ctrl)
modelo4

# MARS: Multivariate Adaptative Regresion Splines

marsGrid <- expand.grid(degree= c(1,2,3), nprune=c(2,5,10,15))

modelo5 <- train(inf_sub~., data=data, method="earth", tuneGrid=marsGrid, trControl=ctrl)
modelo5

# Boosting 

modelo6 <- train(inf_sub~., data=data, method="xgbTree", trControl=ctrl)
modelo6

# Boosting con Grid
boostingGrid <- expand.grid(eta=c(0.01, 0.05, 0.1, 0.2, 0.5, 1), max_depth=c(1,3,5,10), nrounds=c(20,100,500,1000,2000), subsample=c(0.5,0.75,1.0), 
                            colsample_bytree=1, min_child_weight=1, gamma=0)

modelo7 <- train(inf_sub~., data=data, method="xgbTree", tuneGrid=boostingGrid, trControl=ctrl)
modelo7

# PCA
modelo8 <- train(inf_sub~., data=data, method="pcr", trControl=ctrl)
modelo8


# LASSO
modelo9 <- train(inf_sub~., data=data, method="lasso", trControl=ctrl)
modelo9


# THE BAYESIAN LASSO
modelo10 <- train(inf_sub~., data=data, method="blasso", trControl=ctrl)
modelo10

# Ridge Regression
modelo11 <- train(inf_sub~., data=data, method="ridge", trControl=ctrl)
modelo11

# Supervised Principal Component Analysis
modelo12 <- train(inf_sub~., data=data, method="superpc", trControl=ctrl)
modelo12

# Elasticnet
modelo13 <- train(inf_sub~., data=data, method="enet", trControl=ctrl)
modelo13

# Neural Network 
modelo14 <- train(inf_sub~., data=data, method="nnet", trControl=ctrl)
modelo14

# Neural Network Bayesian
modelo15 <- train(inf_sub~., data=data, method="brnn", trControl=ctrl)
modelo15


# Pronosticos

p1 <- predict(modelo1, newdata= data_com)
p1

p2a <- predict(modelo2a, newdata= data_com)
p2a

p2b <- predict(modelo2b, newdata= data_com)
p2b

p2c <- predict(modelo2c, newdata= data_com)
p2c

p3a <- predict(modelo3a, newdata = data_com)
p3a

p3b <- predict(modelo3b, newdata = data_com)
p3b

p4 <- predict(modelo4, newdata = data_com)
p4

p5 <- predict(modelo5, newdata = data_com)
p5

p6 <- predict(modelo6, newdata = data_com)
p6

p7 <- predict(modelo7, newdata = data_com)
p7

p8 <- predict(modelo8, newdata = data_com)
p8

p9 <- predict(modelo9, newdata = data_com)
p9

p10 <- predict(modelo10, newdata = data_com)
p10

p11 <- predict(modelo11, newdata = data_com)
p11

p12 <- predict(modelo12, newdata = data_com)
p12

p13 <- predict(modelo13, newdata = data_com)
p13

p14 <- predict(modelo14, newdata = data_com)
p14

p15 <- predict(modelo15, newdata = data_com)
p15


data_inflacion_final <- cbind(y_var_completo, p1, p2a, p2b, p2c, p3a, p3b, p4, p5, p6, p7, p8, 
                              p9, p10, p11, p12, p13, p14, p15)


#Vamos a exportar todos los resultados 

library(openxlsx)
predic_inflacion <- write.xlsx(data_inflacion_final,".xlsx")
saveWorkbook(predic_inflacion, file = "D:/Usuario/Desktop/predic_inflacion.xlsx", overwrite = TRUE)











