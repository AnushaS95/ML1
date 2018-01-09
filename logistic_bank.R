library(ggplot2)

# Read bank data 
bank <-read.table("C:/Users/yy763/Desktop/CIS490/final_project/Datasets/bank_marketing/bank_additional/bank-additional-full.csv", header = TRUE, sep = ";")
head(bank)
str(bank)

# Visualization of bank dataset with respect to outcome
df_class  <- data.frame(outcome= names(table(bank$y)), count = as.vector(table(bank$y)))
percent <- 100*(df_class$count /sum(df_class$count))
percent <- format(round(percent, 2), nsmall = 2)
p.bar <- ggplot(data= df_class , aes(y = count, x = outcome)) + geom_bar(stat="identity",aes(fill = outcome))+
  geom_text(aes(label=paste0(percent,"%"),y=count), size=4, vjust = -.5) +ylab("Count") + xlab("Outcome variable")  + 
  theme(axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12))+ theme_minimal() 
pdf('barplot_outcome_bank.pdf',height = 6, width = 4)
print(p.bar)
dev.off()

#Set reference factor for outcome variable that we want to predict
bank$y <- relevel(bank$y, ref = "yes")


# Apply logistic regression model for bank dataset:
library(glmnet)

# Divide dataset into training and test dataset
set.seed(400)
sample.ind <- sample(2, 
                     nrow(bank),
                     replace = T,
                     prob = c(0.6,0.4))
train_bank <- bank[sample.ind==1,]
test_bank <- bank[sample.ind==2,]

y_train <- train_bank$y
x_train <- model.matrix(y~., data = train_bank)[,-1]

y_test <- test_bank$y
x_test <- model.matrix(y~., data= test_bank)[,-1]
library(doParallel)
registerDoParallel(10)

# Fitting multinomial regression model
set.seed(1120)
cvfit1 = cv.glmnet(x_train, y_train, family = "binomial",parallel = TRUE, alpha =1)
cvfit0.75=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "binomial",parallel = TRUE, alpha =0.75)
cvfit0.5=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "binomial",parallel = TRUE, alpha =0.5)
cvfit0.25=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "binomial",parallel = TRUE, alpha =0.25)
cvfit0 =cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "binomial",parallel = TRUE, alpha =0)
plot(cvfit0.75)
# Coefficients
coef(cvfit1,s = "lambda.min")
coef(cvfit0.75,s = "lambda.min")
coef(cvfit0.5,s = "lambda.min")
coef(cvfit0.25,s = "lambda.min")
coef(cvfit0,s = "lambda.min")
pdf('Elastic_net_bank.pdf',height = 6, width = 4)
plot(log(cvfit1$lambda),cvfit1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cvfit1$name)
points(log(cvfit0.75$lambda),cvfit0.5$cvm,pch=19,col="grey")
points(log(cvfit0.5$lambda),cvfit0.5$cvm,pch=19,col="blue")
points(log(cvfit0.25$lambda),cvfit0.5$cvm,pch=19,col="orange")
points(log(cvfit0$lambda),cvfit0$cvm,pch=19,col="purple")
legend("topleft",legend=c("alpha= 1(LASSO)","alpha = .75","alpha = .5","alpha = .25","alpha = 0(Ridge)"),pch=19,col=c("red","grey","blue","orange","purple"),bty = "n" )

dev.off()
min(cvfit1$cvm)
min(cvfit0.75$cvm)
min(cvfit0.5$cvm)
min(cvfit0.25$cvm)
min(cvfit0$cvm)
# Prediction
y_pred1 <- predict(cvfit1, newx = x_test, s = "lambda.min", type = "class")
y_pred1
y_pred0.75 <- predict(cvfit0.75, newx = x_test, s = "lambda.min", type = "class")
y_pred0.5 <- predict(cvfit0.5, newx = x_test, s = "lambda.min", type = "class")
y_pred0.25 <- predict(cvfit0.25, newx = x_test, s = "lambda.min", type = "class")
y_pred0 <- predict(cvfit0, newx = x_test, s = "lambda.min", type = "class")
library(caret)
cm1 <- confusionMatrix(factor(y_pred1, levels = levels(y_train)),test_bank$y)
cm1
cm0.75 <- confusionMatrix(factor(y_pred0.75, levels = levels(y_train)),test_bank$y)
cm0.75
cm0.5 <- confusionMatrix(factor(y_pred0.5, levels = levels(y_train)),test_bank$y)
cm0.5
cm0.25 <- confusionMatrix(factor(y_pred0.25, levels = levels(y_train)),test_bank$y)
cm0.25
cm0 <- confusionMatrix(factor(y_pred0, levels = levels(y_train)),test_bank$y)
cm0
cvfit1$lambda.min
cm0$overall["Accuracy"]
cm0$table
cm1$byClass
cm0.75$byClass
cm0.5$byClass
cm0.25$byClass
cm0$byClass

lasso.coef<-predict(cvfit1,type = "coefficients",s = "lambda.min")[1:54,]
lasso.coef


lasso.coef <- data.frame(t(lasso.coef))
lasso.coef <- data.frame(t(lasso.coef))
lasso.coef

lasso.coef <- cbind(rownames(lasso.coef), lasso.coef)
rownames(lasso.coef) <- NULL
colnames(lasso.coef) <- c("variable","value")
lasso.coef


pdf('Coefficients_bank1.pdf',height = 6, width = 10)
coeff.bar <- ggplot(lasso.coef, aes(x = variable, y = value))+  geom_bar(stat = "identity", position = position_dodge())+
  theme_minimal()+
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12))+ 
  xlab("") + ylab("")

print(coeff.bar)
dev.off()

cm_parameters <- data.frame(cm1$byClass)
cm_parameters <- cbind(rownames(cm_parameters), cm_parameters)
rownames(cm_parameters) <- NULL
cm_parameters
cm_parameters <- cm_parameters[1:4,]
cm_parameters
colnames(cm_parameters) <- c("variable","value")
cm_parameters

pdf('confusion_matrix_bank.pdf',height = 6, width = 4)
cm.bar <- ggplot(cm_parameters, aes(x = variable, y = value))+  geom_bar(stat = "identity", position = position_dodge())+
  theme_minimal()+
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12))+ 
  xlab("") + ylab("")
print(cm.bar)
dev.off()

library("pROC")
library("plyr")
library(ROCR)
y_pred <- predict(cvfit1, newx = x_test, s = "lambda.min", type = "response")
head(y_pred)
ROCRPred = prediction(1-y_pred, y_test)
ROCRPref <- performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize = TRUE, print.cutoffs.at=seq(0.1,by=0.1))

y_pred <- predict(cvfit0, newx = x_test, s = "lambda.min", type = "response")
head(y_pred)
ROCRPred = prediction(1-y_pred, y_test)
ROCRPref <- performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize = TRUE, print.cutoffs.at=seq(0.1,by=0.1))

library(pROC)

modelroc=roc(y_test,y_pred)
plot(modelroc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="blue",print.thres=TRUE)
Predictedvalue <- ifelse(y_pred>0.913, "no", "yes")
cm <- confusionMatrix(factor(Predictedvalue, levels = levels(y_train)),test_bank$y)
cm

cm_parameters <- data.frame(cm$byClass)
cm_parameters <- cbind(rownames(cm_parameters), cm_parameters)
rownames(cm_parameters) <- NULL
cm_parameters
cm_parameters <- cm_parameters[1:4,]
cm_parameters
colnames(cm_parameters) <- c("variable","value")
library(reshape2)
pdf('confusion_matrix_bank1_logistic.pdf',height = 6, width = 4)
cm.bar <- ggplot(cm_parameters, aes(x = variable, y = value))+  geom_bar(stat = "identity", position = position_dodge())+
  theme_minimal()+
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12))+ 
  xlab("") + ylab("")
print(cm.bar)
dev.off()


