# install library
install.packages('neuralnet')
# Load library
library(caret)
library(dplyr)
library(neuralnet)
library(RSNNS)
library(rpart)
library(rpart.plot)
# Import dataset
data = read.csv("C:/Users/Wu/Desktop/uOttawa/MDTI/DTI6220[A] Data Analytics & Bus Intell 20219/Project/australia_rain_tomorrow_raw.csv/australia_rain_tomorrow_raw.csv", header=T)

# Show structure and sum up NA rows
str(data)
colSums(is.na(data))


# Select appropriate columns
cleanData <- data %>% select(c(MinTemp,MaxTemp,Rainfall,
                               Temp9am,RainToday,RainTomorrow))
head(cleanData)
# Remove NA rows
data_na_omit <- na.omit(cleanData)
head(data_na_omit)

# Split dataset into 60% train set and 40% test set
samplesize = 0.60 * nrow(data_na_omit)
set.seed(121) #to generate same random sample every time & maintain consistency
index = sample(seq_len (nrow(data_na_omit)), size = samplesize)

# Create training and test set
datatrain = data_na_omit[index,]
datatest = data_na_omit[-index,]
head(datatrain)

# Decision tree model
set.seed(42)
model <- rpart(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + Temp9am + 
                 RainToday, data = datatrain, method = "class") #specify method as class since we are dealing with classification
model
prunedModel <- prune(model, cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
prunedModel
# Plot the model
rpart.plot(model)

# Model evaluation
head(datatest)
head(datatest[,c(1:5)])
preds <- predict(prunedModel, newdata = datatest[,c(1:5)], type = "class") #use the predict() function and pass in the testing subset
preds

# Print the confusion Matrix
table(datatest$RainTomorrow, preds)


# Normalized data
head(data_na_omit)
data_na_omit_numeric <- data_na_omit[,c(-5,-6)]
head(data_na_omit_numeric) 
max = apply(data_na_omit_numeric, 2 , max)
min = apply(data_na_omit_numeric, 2 , min)
scaled = as.data.frame(scale(data_na_omit_numeric, center = min, scale = max - min))
head(scaled)
data_na_omit_normalized <- cbind(scaled,data_na_omit[,c(5,6)])
head(data_na_omit_normalized)

data_na_omit_normalized$RainToday <- ifelse(data_na_omit_normalized$RainToday == "Yes", 1,0)
#data_na_omit_normalized$RainToday
data_na_omit_normalized$RainTomorrow <- ifelse(data_na_omit_normalized$RainTomorrow == "Yes", 1, 0)
#data_na_omit_normalized$RainTomorrow
final_data <- data_na_omit_normalized
head(final_data,10)
nrow(final_data)

# Creating training and test set
trainNN = final_data[index, ]
testNN = final_data[-index, ]
head(trainNN)

# Train MLP
set.seed(15)
head(testNN)

model <- mlp(trainNN[c(1:5)], trainNN[c(6)], size=5, inputsTest=testNN[c(1:5)], targetsTest=testNN[c(6)],linOut=F) 
model
summary(model)

# Prediction
predictions = predict(model,testNN[,c(1:5)])
predictions

# Plot errors and ROC curves
plotIterativeError(model)
plotRegressionError(predictions,testNN[,c(6)])
plotROC(fitted.values(model),trainNN[,c(6)])
plotROC(predictions,testNN[,c(6)])

# Evaluate model
results <- data.frame(actual = testNN[,c(6)], prediction = predictions)
head(results)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
roundedresultsdf
attach(roundedresultsdf)
confusionMatrix(actual, prediction)
45147/55868








## train examples

NN <- mlp(trainNN[c(1:5)], trainNN[c(6)], size=5, learnFunc="BackpropBatch", learnFuncParams=c(10, 0.1), maxit=100, inputsTest=testNN[c(1:5)], targetsTest=testNN[c(6)])
library(Rcpp)
library(nnet)
library(clusterGeneration)
library(nnet)
library(devtools)
plot(NN)

head(testNN)
head(testNN[,c(1:5)])
predict_testNN = predict(NN, testNN[,c(1:5)])
predict_testNN
confusionMatrix(testNN[,c(6)],predict_testNN)
# Results evaluation



results <- data.frame(actual = testNN$RainTomorrow, prediction = predict_testNN$net.result)
head(results)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
confusionMatrix(table(actual,prediction))


NN
#fit a decision tree model and use k-fold CV to evaluate performance
NN <- train(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + Temp9am + 
              RainToday, data = trainNN, method = "mlp")

set.seed(12)
NN = neuralnet(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + Temp9am + 
                 RainToday, trainNN, hidden = 3 , stepmax = 20000, linear.output = FALSE )

# Plot neural network
plot(NN)

# Prediction using neural network
head(testNN)
head(testNN[,c(1:5)])
predict_testNN = compute(NN, testNN[,c(1:5)])

# Results evaluation
results <- data.frame(actual = testNN$RainTomorrow, prediction = predict_testNN$net.result)
head(results)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
confusionMatrix(table(actual,prediction))

# Decision tree model
library(rpart)
library(rpart.plot)

set.seed(42)
model <- rpart(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + Temp9am + 
                 RainToday, data = trainNN, method = "class") #specify method as class since we are dealing with classification
model
prunedModel <- prune(model, cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
prunedModel
# Plot the model
rpart.plot(model)

# Model evaluation
head(testNN)
head(testNN[,c(1:5)])
preds <- predict(prunedModel, newdata = testNN[,c(1:5)], type = "class") #use the predict() function and pass in the testing subset
preds

# Print the confusion Matrix
confusionMatrix(table(testNN$RainTomorrow, preds))

