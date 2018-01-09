library('ggplot2')
library('caret')
# Read bank data 
bank_data <-  read.table("Datasets/bank_marketing/bank_additional/bank-additional-full.csv",sep = ";",stringsAsFactors = TRUE, header = T)
str(bank_data)

library(doParallel)
registerDoParallel(10)


library(nnet)

# Divide dataset into training and test dataset
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




table(train_bank$class)
table(test_bank$class)

# Set up formula
#Encode as a one hot vector multilabel data
train_data <- cbind(x_train, class.ind(as.factor(y_train)))
test_data <- cbind(x_test, class.ind(as.factor(y_test)))

TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=10,savePredictions = T)

NNModel <- train(x_train, y_train,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit
)

NNPredictions <-predict(NNModel,x_test)
cmNN <-confusionMatrix(NNPredictions, y_test)
print(cmNN)

cm_parameters <- data.frame(cmNN$byClass)
#cm_parameters <- cm_parameters[,1:2]
cm_parameters$Class <- gsub("Class: ","",rownames(cm_parameters))



library(reshape2)
cm_parameters_m <- melt(cm_parameters,id.vars = c("Class"))
#pdf('confusion_matrix_param_nn.pdf',height = 6, width = 4)
cm.bar <- ggplot(cm_parameters_m, aes(x = variable, y = value, fill = Class ))+  geom_bar(stat = "identity", position = position_dodge())+
  theme_minimal()+
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12))+ 
  xlab("") + ylab("")
print(cm.bar)
#dev.off()



pdf("Accuracy_NN.pdf",height = 6, width = 10)
plot(NNModel)
dev.off()


varImp(NNModel,scale = F)
