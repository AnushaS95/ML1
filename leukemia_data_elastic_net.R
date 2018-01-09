
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


# Apply multinomial logistic regression model for car dataset:
library(glmnet)
set.seed(1120)
mat_train<-as.matrix(as.data.frame(leukemia_data_train))
mat2<-scale(t(mat_train), scale = TRUE, center = TRUE)
mat_train<-t(mat2)

mat_test<-as.matrix(as.data.frame(leukemia_data_test))
mat2<-scale(t(mat_test), scale = TRUE, center = TRUE)
mat_test<-t(mat2)

x_train <- mat_train

x_test <-  mat_test

x_train <- as.matrix(leukemia_data_train)

x_test <-  as.matrix(leukemia_data_test)


library(doParallel)
registerDoParallel(10)

# Fitting multinomial regression model
# Fitting multinomial regression model
library("glmnet")
set.seed(1120)
#lambda = cvfit0.75$lambda,
cvfit1 = cv.glmnet(x_train, y_train,family = "binomial",parallel = TRUE, alpha =1, type.measure = "class")
cvfit0.75=cv.glmnet(x_train, y_train, family = "binomial",parallel = TRUE, alpha =0.75,type.measure = "class")
cvfit0.5=cv.glmnet(x_train, y_train,family = "binomial",parallel = TRUE, alpha =0.5,type.measure = "class")
cvfit0.25=cv.glmnet(x_train, y_train, family = "binomial",parallel = TRUE, alpha =0.25,type.measure = "class")
cvfit0 =cv.glmnet(x_train, y_train, family = "binomial",parallel = TRUE, alpha =0,type.measure = "class")
plot(cvfit0.75)


coef(cvfit0.75,s = "lambda.min")
coef(cvfit0.5,s = "lambda.min")
coef(cvfit0.25,s = "lambda.min")
coef(cvfit0,s = "lambda.min")


pdf('Elastic_net_leukemia.pdf',height = 6, width = 6)
#par(mfrow=c(3,3))
#plot(cvfit1);plot(cvfit0.75);plot(cvfit0.5);plot(cvfit0.25);plot(cvfit0)
plot(log(cvfit1$lambda),cvfit1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cvfit1$name,ylim=c(0,0.4))
points(log(cvfit0.75$lambda),cvfit0.75$cvm,pch=19,col="grey")
points(log(cvfit0.5$lambda),cvfit0.5$cvm,pch=19,col="blue")
points(log(cvfit0.25$lambda),cvfit0.25$cvm,pch=19,col="orange")
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
y_pred1 <- predict(cvfit1, newx = x_test, s = "lambda.1se", type = "response")
y_pred0.75 <- predict(cvfit0.75, newx = x_test, s = "lambda.1se", type = "response")
y_pred0.5 <- predict(cvfit0.5, newx = x_test, s = "lambda.1se", type = "response")
y_pred0.25 <- predict(cvfit0.25, newx = x_test, s = "lambda.1se", type = "response")
y_pred0 <- predict(cvfit0, newx = x_test, s = "lambda.1se", type = "response")


library(pROC)

perf1 <- roc( y_test,as.numeric(y_pred1),smooth = TRUE)
perf1 <- smooth(perf1,method = "density")
perf1 <-  data.frame(tpr=perf1$sensitivities,fpr= (1 - perf1$specificities), alpha = 1,model="GLM")

perf0.75 <- roc( y_test,as.numeric(y_pred0.75))
perf0.75 <- smooth(perf0.75,method = "density")
perf0.75 <- data.frame(tpr=perf0.75$sensitivities,fpr= (1 - perf0.75$specificities),  alpha = 0.75,model="GLM")

perf0.5 <- roc( y_test,as.numeric(y_pred0.5))
perf0.5 <- smooth(perf0.5,method = "density")
perf0.5 <-  data.frame(tpr=perf0.5$sensitivities,fpr= (1 - perf0.5$specificities), alpha = 0.5, model="GLM")

perf0.25 <- roc( y_test,as.numeric(y_pred0.25))
perf0.25 <- smooth(perf0.25,method = "density")
perf0.25 <-  data.frame(tpr=perf0.25$sensitivities,fpr= (1 - perf0.25$specificities), alpha = 0.25, model="GLM")

perf0 <- roc( y_test,as.numeric(y_pred0),smooth = T)
perf0 <- smooth(perf0,method = "density")
perf0 <-  data.frame(tpr=perf0$sensitivities,fpr= (1 - perf0$specificities), alpha = 0, model="GLM")

l <- list(perf0,perf0.25,perf0.5,perf0.75,perf1)
#l <- list(perf0.75,perf1)
roc.data <-  do.call(rbind, l)
roc.data$alpha <- factor(roc.data$alpha)

library("ggplot2")
pdf('ROC_curve_elastic_net_leukemia.pdf',height = 6, width = 6)
roc_curve <- ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr,color = alpha)) +
  #geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr),size = 2) +
  geom_abline(aes(intercept = 0 ,slope = 1),linetype="dashed")+
  theme_bw()+
  scale_y_continuous(expand = c(0, 0))+
  scale_x_continuous(expand = c(0, 0))+
  scale_color_manual(values = rev(c("red","grey","blue","orange","purple")))+
  guides(colour = guide_legend(override.aes = list(size=3)))+
  ylab("Sensitivity")+
  xlab("False Positive Rate")+
  coord_fixed(ratio = 1)
  #ggtitle(paste0("ROC Curve w/ AUC=", perf1$auc))
print(roc_curve)
dev.off()

# Prediction
y_pred1 <- predict(cvfit1, newx = x_test, s = "lambda.1se", type = "class")
y_pred0.75 <- predict(cvfit0.75, newx = x_test, s = "lambda.1se", type = "class")
y_pred0.5 <- predict(cvfit0.5, newx = x_test, s = "lambda.1se", type = "class")
y_pred0.25 <- predict(cvfit0.25, newx = x_test, s = "lambda.1se", type = "class")
y_pred0 <- predict(cvfit0, newx = x_test, s = "lambda.1se", type = "class")

library(caret)
cm1 <- confusionMatrix(factor(y_pred1, levels = levels(y_train)),y_test)
cm1
cm0.75 <- confusionMatrix(factor(y_pred0.75, levels = levels(y_train)),y_test)
cm0.75
cm0.5 <- confusionMatrix(factor(y_pred0.5, levels = levels(y_train)),y_test)
cm0.5
cm0.25 <- confusionMatrix(factor(y_pred0.25, levels = levels(y_train)),y_test)
cm0.25
cm0 <- confusionMatrix(factor(y_pred0, levels = levels(y_train)),y_test)
cm0

# Coefficients
co <- coef(cvfit0.25,s = "lambda.1se")
inds<-which(co !=0)
variables<-row.names(co)[inds]
variables<-variables[!(variables %in% '(Intercept)')];
variables

train_subset <-  leukemia_data_train[,variables]
test_subset <-  leukemia_data_test[,variables]

library(viridis)
library("ComplexHeatmap")
sig_data_c<-data.frame(t(train_subset))
genotype_colors<-rep("darkred",length(y_train))
genotype_colors[which(y_train=="AML")]<-"darkblue"
dist.pear<-function(x) as.dist(1-cor(t(x)))
mat<-as.matrix(as.data.frame(sig_data_c))
mat2<-scale(t(mat), scale = TRUE, center = TRUE)
mat2<-t(mat2)
mat2[mat2==0]<-NA
annotations<-data.frame(as.character(y_train))
names(annotations)<-c("Type")
ha_column = HeatmapAnnotation(annotations,col = list(Type = c("AML" =  "red", "ALL" = "blue")))
pdf('Train_subset_heatmap_elastic_net.pdf',height = 10, width = 10)
ht1 = Heatmap(mat2, name = "Scale", column_title = "AML vs ALL", top_annotation = ha_column, clustering_distance_rows = "spearman",
              clustering_method_rows = "ward.D2",row_names_side = "left", km=1,col=viridis(10), row_dend_side="right",
              show_column_names = "FALSE", width=4, row_names_max_width = unit(8, "cm"),row_names_gp = gpar(fontsize = 9), cluster_columns = FALSE,
              na_col="white")
ht_list = ht1
draw(ht_list)
dev.off()


test_subset 
order_type <- c(which(y_test == "ALL"),which(y_test == "AML"))
test_subset <-  test_subset[order_type,]
y_test1 <- y_test[order_type]

sig_data_c<-data.frame(t(test_subset))
genotype_colors<-rep("darkred",length(y_test1))
genotype_colors[which(y_train=="AML")]<-"darkblue"
dist.pear<-function(x) as.dist(1-cor(t(x)))
mat<-as.matrix(as.data.frame(sig_data_c))
mat2<-scale(t(mat), scale = TRUE, center = TRUE)
mat2<-t(mat2)
mat2[mat2==0]<-NA
annotations<-data.frame(as.character(y_test1))
names(annotations)<-c("Type")
ha_column = HeatmapAnnotation(annotations,col = list(Type = c("AML" =  "red", "ALL" = "blue")))
pdf('Test_subset_heatmap_elastic_net.pdf',height = 10, width = 10)
ht1 = Heatmap(mat2, name = "Scale", column_title = "AML vs ALL", top_annotation = ha_column, clustering_distance_rows = "spearman",
              clustering_method_rows = "ward.D2",row_names_side = "left", km=1,col=viridis(10), row_dend_side="right",
              show_column_names = "FALSE", width=4, row_names_max_width = unit(8, "cm"),row_names_gp = gpar(fontsize = 9), cluster_columns = FALSE,
              na_col="white")
ht_list = ht1
draw(ht_list)
dev.off()

cm_parameters <- data.frame(cm0.25$byClass)
#cm_parameters <- cm_parameters[1:4,]
pdf('confusion_matrix_param_elastic_net.pdf',height = 6, width = 6)
par(mar=c(3, 10, 3, 1))
barplot(cm_parameters[1:4,],names.arg = rownames(cm_parameters)[1:4],las=2,cex.names=1,horiz = TRUE,col = "blue")
dev.off()

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

