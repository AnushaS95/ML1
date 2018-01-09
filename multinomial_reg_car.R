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

# Apply multinomial logistic regression model for car dataset:
library(glmnet)

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


library(doParallel)
registerDoParallel(10)

# Fitting multinomial regression model
set.seed(1120)
cvfit1 = cv.glmnet(x_train, y_train, family = "multinomial",parallel = TRUE, alpha =1)
cvfit0.75=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "multinomial",parallel = TRUE, alpha =0.75)
cvfit0.5=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "multinomial",parallel = TRUE, alpha =0.5)
cvfit0.25=cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "multinomial",parallel = TRUE, alpha =0.25)
cvfit0 =cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "multinomial",parallel = TRUE, alpha =0)
cvfit0 =cv.glmnet(x_train, y_train,lambda = cvfit1$lambda, family = "multinomial",parallel = TRUE, alpha =0)
plot(cvfit0.75)
# Coefficients
coef(cvfit1,s = "lambda.min")
coef(cvfit0.75,s = "lambda.min")
coef(cvfit0.5,s = "lambda.min")
coef(cvfit0.25,s = "lambda.min")
coef(cvfit0,s = "lambda.min")


pdf('Elastic_net.pdf',height = 6, width = 4)
#par(mfrow=c(3,3))
#plot(cvfit1);plot(cvfit0.75);plot(cvfit0.5);plot(cvfit0.25);plot(cvfit0)
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

plot(cvfit1,xvar="lambda",label=TRUE)


# Prediction
y_pred1 <- predict(cvfit1, newx = x_test, s = "lambda.min", type = "class")
y_pred0.75 <- predict(cvfit0.75, newx = x_test, s = "lambda.min", type = "class")
y_pred0.5 <- predict(cvfit0.5, newx = x_test, s = "lambda.min", type = "class")
y_pred0.25 <- predict(cvfit0.25, newx = x_test, s = "lambda.min", type = "class")
y_pred0 <- predict(cvfit0, newx = x_test, s = "lambda.min", type = "class")


c_y <- data.frame(y_pred = factor(y_pred, levels = levels(y_train)),y_real = y_test)


library(caret)
cm1 <- confusionMatrix(factor(y_pred1, levels = levels(y_train)),test_car$class)
cm1
cm0.75 <- confusionMatrix(factor(y_pred0.75, levels = levels(y_train)),test_car$class)
cm0.75
cm0.5 <- confusionMatrix(factor(y_pred0.5, levels = levels(y_train)),test_car$class)
cm0.5
cm0.25 <- confusionMatrix(factor(y_pred0.25, levels = levels(y_train)),test_car$class)
cm0.25
cm0 <- confusionMatrix(factor(y_pred0, levels = levels(y_train)),test_car$class)
cm0


cvfit1$lambda.min
cm0$overall["Accuracy"]
cm1$table

rownames(coef(cvfit1,s = "lambda.min")$acc)

coefficients_glm <-  sapply(coef(cvfit1,s = "lambda.min"),as.matrix)[-1,]
rownames(coefficients_glm) <- rownames(coef(cvfit1,s = "lambda.min")$acc)[-1]
coefficients_glm <- data.frame(t(coefficients_glm))
coefficients_glm$Class <- rownames(cm_parameters)

coefficients_glm_m <- melt(coefficients_glm,id.vars = c("Class"))
pdf('Coefficients_glm.pdf',height = 6, width = 10)
coeff.bar <- ggplot(coefficients_glm_m, aes(x = variable, y = value, fill = Class ))+  geom_bar(stat = "identity", position = position_dodge())+
  facet_wrap(~Class,scales = "free_y")+theme_minimal()+ 
  theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
        axis.text.y=element_text(size=12),
        strip.text.x = element_text(size = 10))+ 
  xlab("") + ylab("")
print(coeff.bar)
dev.off()




cm_parameters <- data.frame(cm1$byClass)
cm_parameters <- cm_parameters[,1:4]
cm_parameters$Class <- gsub("Class: ","",rownames(cm_parameters))



library(reshape2)
cm_parameters_m <- melt(cm_parameters,id.vars = c("Class"))
pdf('confusion_matrix_param.pdf',height = 6, width = 4)
cm.bar <- ggplot(cm_parameters_m, aes(x = variable, y = value, fill = Class ))+  geom_bar(stat = "identity", position = position_dodge())+
          theme_minimal()+
          theme(axis.text.x=element_text(size=12,angle = 90, vjust = 0.5),
                axis.text.y=element_text(size=12))+ 
          xlab("") + ylab("")
print(cm.bar)
dev.off()



library("pROC")
install.packages("gdata", dependencies=TRUE)
library("plyr")
sen_spec1 <-  data.frame(cm1$byClass[,1:2])
sen_spec0.75 <-  data.frame(cm0.75$byClass[,1:2])
sen_spec0.5 <-  data.frame(cm0.5$byClass[,1:2])
sen_spec0.25 <-  data.frame(cm0.25$byClass[,1:2])
sen_spec0 <-  data.frame(cm0$byClass[,1:2])

concat_data <-rbind.fill(sen_spec1, sen_spec0.75, sen_spec0.5,sen_spec0.25,sen_spec0)

concat_data$class <- gsub("Class: ","",rep(rownames(sen_spec1),5))


concat_data_unacc <- concat_data[concat_data$class == "unacc",]

# sensitivity 
sens_unacc <- concat_data_unacc$Sensitivity

# Specificity
spec_unacc <-   concat_data_unacc$Specificity

# True Positive Rate
tpr <-  sens

# False Positive Rate
fpr <-  1- spec

# Generate a roc object with any values and
# add the given sensitivities and specificities
roc_cur <- roc(c(1,0,0),c(1,1,0))

roc_cur$sensitivities <-  sens
roc_cur$specificities <- spec

plot.roc(smooth(roc_cur),legacy.axes = TRUE,col ="blue")


