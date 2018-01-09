library('ggplot2')
library('caret')
library('randomForest')
# Read car data 
car_data <-  read.table("Datasets/cars/car.data",sep = ",",stringsAsFactors = TRUE)
names(car_data) <-  c("buying","maint","doors","persons","lug_boot","safety","class")
str(car_data)

#Set reference factor for class variable that we want to predict
car_data$class <- relevel(car_data$class, ref = "unacc")

set.seed(400)
sample.ind <- sample(2, 
                     nrow(car_data),
                     replace = T,
                     prob = c(0.6,0.4))
train_car <- car_data[sample.ind==1,]
test_car <- car_data[sample.ind==2,]

y_train <- train_bank[,7]
x_train <- model.matrix(~ buying+maint+doors+persons+lug_boot+safety-1,data=train_car)
y_test <- test_bank[,7]
x_test <- model.matrix(~ buying+maint+doors+persons+lug_boot+safety-1,data=test_car)

TrainingParameters <- trainControl(method = "oob", savePredictions = T)

RFModel <- train(x_train, y_train,
                 method = "rf",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 importance=T,
                 ntree=500
)

ggplot(RFModel)
print(unclass(RFModel)$finalModel)

plot(unclass(RFModel)$finalModel, main='Classifier Performance')
varImpPlot(unclass(RFModel)$finalModel, main="Variable Importance Plot")
RFPredictions <- predict(unclass(RFModel)$finalModel,x_test)
cmRF <-confusionMatrix(RFPredictions, y_test)
print(cmRF)
