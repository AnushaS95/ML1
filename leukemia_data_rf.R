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

x_train <- leukemia_data_train
x_test <- leukemia_data_test


x_train <- model.matrix(~.-1,data=train_lkma)
x_test <- model.matrix(~.-1,data=test_lkma)


# library("neuralnet")
# library("caret")
# library("nnet")
# 
# 
# library(parallel)
# library(doParallel)
# cluster <- makeCluster(10) # convention to leave 1 core for OS
# registerDoParallel(cluster)


# Set up formula
#Encode as a one hot vector multilabel data
train_data <- x_train
# Set labels name
names <-  colnames(train_data)

f <- as.formula(paste("y_trainALL + y_trainAML ~", paste(names[!names %in% c("y_trainALL","y_trainAML")], collapse = " + ")))
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


set.seed(1120)
mat_train<-as.matrix(as.data.frame(leukemia_data_train))
mat2<-scale(t(mat_train), scale = TRUE, center = TRUE)
mat_train<-t(mat2)

mat_test<-as.matrix(as.data.frame(leukemia_data_test))
mat2<-scale(t(mat_test), scale = TRUE, center = TRUE)
mat_test<-t(mat2)



library("caret")
TrainingParameters <- trainControl(method = "repeatedcv", number = 3, repeats=3 ,savePredictions = T,classProbs=T)
set.seed(1120)
library(doMC)
registerDoMC(10)
rfModel <- train(leukemia_data_train, y_train,
                 method = "rf",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit,
                 tuneLength=10,
                 allowParallel = TRUE
)

pdf("Accuracy_RF_leukemia.pdf",height = 6, width = 10)
plot(rfModel)
dev.off()

saveRDS(rfModel,"rfModel.rds")

library(pROC)
perf <- list()
for (mtry in  rfModel$results$mtry[5:7]){
  
  # Plot:
  result.roc <- roc(rfModel$pred$obs[selectedIndices],rfModel$pred$ALL[selectedIndices],smooth = FALSE)
  result.roc <- smooth(result.roc,method = "density")
  perf[[mtry]] <- data.frame(tpr=result.roc$sensitivities,fpr= (1 - result.roc$specificities), mtry = mtry,model="rf")
  
}

#l <- list(perf0,perf0.25,perf0.5,perf0.75,perf1)
#l <- list(perf0.75,perf1)
roc.data <-  do.call(rbind, perf)
roc.data$mtry <- factor(roc.data$mtry)

library("ggplot2")
pdf('ROC_curve_rf_leukemia.pdf',height = 6, width = 6)
roc_curve <- ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr,color = mtry)) +
  #geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr),size = 1) +
  geom_abline(aes(intercept = 0 ,slope = 1),linetype="dashed")+
  theme_bw()+
  scale_y_continuous(expand = c(0, 0))+
  scale_x_continuous(expand = c(0, 0))+
  #scale_color_manual(values = rev(c("red","grey","blue","orange","purple")))+
  guides(colour = guide_legend(override.aes = list(size=3)))+
  ylab("Sensitivity")+
  xlab("False Positive Rate")+
  coord_fixed(ratio = 1)+
ggtitle(paste0("ROC Curve w/ AUC=", result.roc$auc))
print(roc_curve)
dev.off()

# # Select a parameter setting
# selectedIndices <- rfModel$pred$mtry == 188
# # Plot:
# result.roc <- roc(rfModel$pred$obs[selectedIndices],
#          rfModel$pred$ALL[selectedIndices])
# result.roc <- smooth(result.roc,method = "density")
# plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

rfModel
rf_pred <-predict(rfModel,as.matrix(leukemia_data_test))
cmrf <-confusionMatrix(rf_pred, y_test)
print(cmrf)

cm_parameters <- data.frame(cmrf$byClass)
#cm_parameters <- cm_parameters[1:4,]
pdf('confusion_matrix_param_rf.pdf',height = 6, width = 6)
par(mar=c(3, 10, 3, 1))
barplot(cm_parameters[1:4,],names.arg = rownames(cm_parameters)[1:4],las=2,cex.names=1,horiz = TRUE,col = "blue")
dev.off()





# Variable Importance
var_imp <- data.frame(rfModel$finalModel$importance)
var_imp$Variables <- rownames(var_imp)
var_imp <-var_imp[order(var_imp$MeanDecreaseGini,decreasing = T),][1:50,]
variables <- var_imp$Variables

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
pdf('Train_subset_heatmap_rf.pdf',height = 10, width = 10)
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
pdf('Test_subset_heatmap_rf.pdf',height = 10, width = 10)
ht1 = Heatmap(mat2, name = "Scale", column_title = "AML vs ALL", top_annotation = ha_column, clustering_distance_rows = "spearman",
              clustering_method_rows = "ward.D2",row_names_side = "left", km=1,col=viridis(10), row_dend_side="right",
              show_column_names = "FALSE", width=4, row_names_max_width = unit(8, "cm"),row_names_gp = gpar(fontsize = 9), cluster_columns = FALSE,
              na_col="white")
ht_list = ht1
draw(ht_list)
dev.off()










