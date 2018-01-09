# Read car data 
car_data <-  read.table("Datasets/cars/car.data",sep = ",",stringsAsFactors = TRUE)
names(car_data) <-  c("buying","maint","doors","persons","lug_boot","safety","class")
str(car_data)

# Visualization of car dataset with respect to class 
df_class  <- data.frame(class= names(table(car_data$class)), count = as.vector(table(car_data$class)))
percent <- 100*(df_class$count /sum(df_class$count))
percent <- format(round(percent, 2), nsmall = 2)
p.bar <- ggplot(data= df_class , aes(y = count, x = class)) + geom_bar(stat="identity",aes(fill = class))+
  geom_text(aes(label=paste0(percent,"%"),y=count), size=4, vjust = -.5) +ylab("Count") + xlab("Class")  + 
  theme(axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12))+ theme_minimal() 
pdf('barplot_class_car.pdf',height = 6, width = 4)
print(p.bar)
dev.off()

#Set reference factor for class variable that we want to predict
car_data$class <- relevel(car_data$class, ref = "unacc")

library(doParallel)
registerDoParallel(10)

# Apply multinomial logistic regression model for car dataset:
library(nnet)

# Divide dataset into training and test dataset
set.seed(400)
sample.ind <- sample(2, 
                     nrow(car_data),
                     replace = T,
                     prob = c(0.6,0.4))
train_car <- car_data[sample.ind==1,]
test_car <- car_data[sample.ind==2,]

y_train <- train_car[,7]
x_train <- model.matrix(~ buying+maint+doors+persons+lug_boot+safety-1,data=train_car)

y_test <- test_car[,7]
x_test <- model.matrix(~ buying+maint+doors+persons+lug_boot+safety-1,data=test_car)




table(train_car$class)
table(test_car$class)

# 
# mlr_car <- multinom(class~buying+maint+doors+persons+lug_boot+safety, data= train_car)
# summary(mlr_car)
# 
# #calculate the p.value
# z <- summary(mlr_car)$coefficients/summary(mlr_car)$standard.errors
# p <- (1 - pnorm(abs(z), 0, 1))*2
# p
# 
# #Predict
# predict(mlr_car,test_car,type = "class")
# 
# # Misclassification Error
# library(caret)
# cm <- confusionMatrix(predict(mlr_car,test_car,type = "class"),test_car$class)
# cm
# 
# 1 - sum(diag(cm))/sum(cm)
# 
# 
# car_data

library(neuralnet)


# Set up formula
#Encode as a one hot vector multilabel data
train_data <- cbind(x_train, class.ind(as.factor(y_train)))
# Set labels name
names <-  colnames(train_data)

f <- as.formula(paste("unacc + acc + good + vgood ~", paste(names[!names %in% c("unacc","acc","good","vgood")], collapse = " + ")))
f

n <- neuralnet(f,
               data = train_data,
               hidden = 5,
               act.fct = "logistic",
               linear.output = FALSE,
               lifesign = "minimal")

pdf("Neural_Network_hid_5.pdf",height = 6, width = 10)
plot(n)
dev.off()
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=10,savePredictions = T)

NNModel <- train(train_data[,1:16], train_car$class,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)

NNPredictions <-predict(NNModel,x_test)
cmNN <-confusionMatrix(NNPredictions, test_car$class)
print(cmNN)

cm_parameters <- data.frame(cmNN$byClass)
cm_parameters <- cm_parameters[,1:4]
cm_parameters$Class <- gsub("Class: ","",rownames(cm_parameters))



library(reshape2)
cm_parameters_m <- melt(cm_parameters,id.vars = c("Class"))
pdf('confusion_matrix_param_nn.pdf',height = 6, width = 4)
cm.bar <- ggplot(cm_parameters_m, aes(x = variable, y = value, fill = Class ))+  geom_bar(stat = "identity", position = position_dodge())+
  theme_minimal()+
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12))+ 
  xlab("") + ylab("")
print(cm.bar)
dev.off()



pdf("Accuracy_NN.pdf",height = 6, width = 10)
plot(NNModel)
dev.off()


varImp(NNModel,scale = F)



library(pROC)
# Select a parameter setting
selectedIndices <- NNModel$pred$size == 3
# Plot:
multiclass.roc(as.numeric(NNModel$pred[selectedIndices,]$obs),
                    as.numeric(NNModel$pred[selectedIndices,]$pred))


# Compute predictions
output <-  compute(n,test_data[,1:16])
p1 <-  output$net.result

