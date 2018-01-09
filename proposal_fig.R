
# Read car data 
car_data <-  read.table("Datasets/cars/car.data",sep = ",",stringsAsFactors = TRUE)
names(car_data) <-  c("buying","maint","doors","persons","lug_boot","safety","class")
sapply(car_data, class)


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
sapply(car_data,levels)
                          
                          
# Read bank data 
bank_data <-  read.table("Datasets/bank_marketing/bank/bank.csv",sep = ";",stringsAsFactors = TRUE,header = TRUE)
df_job  <- data.frame(table(bank_data$job,bank_data$y))
percent <- 100*(df_class$count /sum(df_class$count))
percent <- format(round(percent, 2), nsmall = 2)
p.bar <- ggplot(data= df_class , aes(y = count, x = class)) + geom_bar(stat="identity",aes(fill = class))+
  geom_text(aes(label=paste0(percent,"%"),y=count), size=4, vjust = -.5) +ylab("Count") + xlab("Class")  + 
  theme(axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12))+ theme_minimal() 
pdf('barplot_class_car.pdf',height = 6, width = 4)
print(p.bar)
dev.off()
sapply(car_data,levels)
table(bank_data$job,bank_data$y)
