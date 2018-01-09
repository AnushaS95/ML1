# Read Leukemia data 
leukemia_data_train <-  read.csv("Datasets/leukemia/data_set_ALL_AML_train.csv",sep = "\t")
gene_names <- leukemia_data_train$Gene.Accession.Number
leukemia_data_train <- leukemia_data_train[,grep("X",colnames(leukemia_data_train))]
leukemia_data_train <- data.frame(t(leukemia_data_train))
colnames(leukemia_data_train) <-  gene_names
rownames(leukemia_data_train) <- gsub("X","",rownames(leukemia_data_train) )
colnames(leukemia_data_train) <- gsub("-","_",colnames(leukemia_data_train))
colnames(leukemia_data_train) <- gsub("\\/","_",colnames(leukemia_data_train))
leukemia_data_train <- leukemia_data_train[ order(as.numeric(row.names(leukemia_data_train))), ]

leukemia_data_test <-  read.csv("Datasets/leukemia/data_set_ALL_AML_independent.csv",sep = "\t")
leukemia_data_test <- leukemia_data_test[,grep("X",colnames(leukemia_data_test))]
leukemia_data_test <- data.frame(t(leukemia_data_test))  
colnames(leukemia_data_test) <-  gene_names
rownames(leukemia_data_test) <- gsub("X","",rownames(leukemia_data_test) )
colnames(leukemia_data_test) <- gsub("-","_",colnames(leukemia_data_test))
colnames(leukemia_data_test) <- gsub("\\/","_",colnames(leukemia_data_test))
leukemia_data_test <- leukemia_data_test[ order(as.numeric(row.names(leukemia_data_test))), ]


Y <- read.csv("Datasets/leukemia/table_ALL_AML_samples.csv",sep = "\t",header = FALSE)
Y$V2 <- trimws(Y$V2, which = c("both"))
y_train <-  factor(Y$V2[1:38])
y_test <-  factor(Y$V2[39:72])
train_lkma <- cbind(leukemia_data_train,y_train)
test_lkma <- cbind(leukemia_data_test,y_test)

x_train <- model.matrix(~.-1,data=train_lkma)
x_test <- model.matrix(~.-1,data=test_lkma)


library("neuralnet")
library("caret")
library("nnet")


library(parallel)
library(doParallel)
cluster <- makeCluster(10) # convention to leave 1 core for OS
registerDoParallel(cluster)



TrainingParameters <- trainControl(method = "repeatedcv", number = 5, repeats=1,savePredictions = T)

NNModel <- train(as.matrix(leukemia_data_train), y_train,
                 method = "nnet",
                 trControl= TrainingParameters,
                 #preProcess=c("scale","center"),
                 na.action = na.omit,
                 tuneLength=5,
                 allowParallel = TRUE,
                 MaxNWts =10000000
)

dt_res <-  data.frame(NNModel$results)
dt_res <- dt_res[1:4,]

ggplot(dt_res,aes(x = decay, y = Accuracy)) + geom_point(shape=21, size = 3) + geom_line(color = "blue")+
   theme_bw() 

NNModel
nn_pred <-predict(NNModel,as.matrix(leukemia_data_test))
cmnn <-confusionMatrix(nn_pred, y_test)
print(cmnn)

set.seed(1120)
library(doMC)
registerDoMC(10)

#nn_model <- nnet(leukemia_data_train, y_train, family="binomial",size = 5574900,softmax=TRUE,)


my.grid <- expand.grid(.decay = c(0.5), .size = c(5))
prestige.fit <- train(y_trainALL + y_trainAML~., data = x_train,
                      method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F, linout = T,
                      allowParallel = TRUE) 
model <- train(y_trainALL + y_trainAML~., data = x_train[,7000:7131], method='nnet', linout=TRUE, trace = FALSE,
               #Grid of tuning parameters to try:
               tuneGrid=expand.grid(.size=c(1,5,10),.decay=c(0,0.001,0.1))) 

library(randomForest)
#rf <- randomForest(x = leukemia_data_train, y = y_train, xtest = leukemia_data_test, ytest = y_test,mtry = 200, ntree = 1500   ,allowParallel = TRUE) 

library("pROC")
# Select a parameter setting
selectedIndices <- NNModel$pred$mtry == 688
# Plot:
plot(roc(NNModel$pred$pred[selectedIndices],
         NNModel$pred$obs[selectedIndices]))

#roc(NNModel$pred$,    NNModel$pred$obs[selectedIndices])

NN_prob <-predict(NNModel,as.matrix(leukemia_data_test),type = "prob")
NN_pred <-predict(NNModel,as.matrix(leukemia_data_test),type = "prob")

plot(roc(predictor=NN_prob$ALL,    response=y_test,,smooth = T))



cmNN <-confusionMatrix(NNPredictions, y_test)
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
