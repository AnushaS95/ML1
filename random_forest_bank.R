library('ggplot2')
library('caret')
library('randomForest')
# Read car data 
bank_data <-  read.table("Datasets/bank_marketing/bank_additional/bank-additional-full.csv",sep = ";",stringsAsFactors = TRUE, header = T)
#names(car_data) <-  c("buying","maint","doors","persons","lug_boot","safety","class")
str(bank_data)

set.seed(400)
sample.ind <- sample(2, 
                     nrow(bank_data),
                     replace = T,
                     prob = c(0.6,0.4))
train_bank <- bank_data[sample.ind==1,]
test_bank <- bank_data[sample.ind==2,]

y_train <- train_bank[,'y']
x_train <- model.matrix(y ~ . - 1,data=train_bank)

y_test <- test_bank[,'y']
x_test <- model.matrix(y ~ . - 1,data=test_bank)

TrainingParameters <- trainControl(method = "oob", savePredictions = T)

RFModel <- train(x_train, y_train,
                 method = "rf",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit,
                 importance=T,
                 ntree=500
)

ggplot(RFModel)

plot(unclass(RFModel)$finalModel, main='Classifier Performance')
varImpPlot(unclass(RFModel)$finalModel, main="Variable Importance Plot")
RFPredictions <- predict(RFModel,x_test)
cmRF <-confusionMatrix(RFPredictions, y_test)
print(cmRF)
