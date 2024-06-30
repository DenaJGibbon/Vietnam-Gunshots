library(tidyverse)

# With balanced Vietnam data
performancetablesA <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_binary_unfrozen_FALSE_/performance_tables/'
performancetablesB <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_binary_unfrozen_TRUE_/performance_tables/'
# 
# # With Belize data
performancetablesC <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_belize_binary_unfrozen_FALSE_/performance_tables/'
performancetablesD <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_belize_binary_unfrozen_TRUE_/performance_tables/'

# With unbalanced Vietnam data
performancetablesE <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnamunbalanced_binary_unfrozen_FALSE_/performance_tables'
performancetablesF <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnamunbalanced_binary_unfrozen_TRUE_/performance_tables'

performancetablesG <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/belizetest/performance_tables_trained/'

# Find best performing model
FrozenFiles <- list.files(c(performancetablesA,performancetablesB,performancetablesC,performancetablesD,
                            performancetablesE,performancetablesF),#performancetablesC),#,performancetablesC,performancetablesD,performancetablesE),
                          full.names = T)

FrozenCombined <- FrozenFiles %>% 
  lapply(read_csv) %>%                              # Store all files in list
  bind_rows                                         # Combine data sets into one data set 

FrozenCombined <- as.data.frame(FrozenCombined)
unique_training_data <- unique(FrozenCombined$`Training Data`)

#FrozenCombined$Threshold <- 1-  FrozenCombined$Threshold
performance_scores <-  FrozenCombined #subset(FrozenCombined, Threshold== 0.5  | Threshold== 0.85  |Threshold== 0.95   )

best_f1_results <- data.frame()
best_precision_results <- data.frame()
best_recall_results <- data.frame()
best_auc_results <- data.frame()
# Loop through each 'TrainingData' type and find the row with the maximum F1 score
for (td in unique_training_data) {
  subset_data <- (subset(performance_scores, `Training Data` == td   ))
  subset_data[,c(1:14)] [is.na(subset_data[,c(1:14)])] <- 0
  subset_data <- subset(subset_data,Threshold >= 0.1)
  max_f1_row <- subset_data[which.max(subset_data$F1), ]
  best_f1_results <- rbind.data.frame(best_f1_results, max_f1_row)
  max_precision_row <- subset_data[which(subset_data$Precision==max(subset_data$Precision)), ]
  
  max_precision_row <- max_precision_row[which.max(max_precision_row$F1), ]
  
  best_precision_results <-rbind.data.frame(best_precision_results, max_precision_row)
  max_recall_row <- subset_data[which.max(subset_data$Recall), ]
  best_recall_results <-rbind.data.frame(best_recall_results, max_recall_row)
  
  max_auc_row <- subset_data[which.max(subset_data$AUC), ]
  best_auc_results <-rbind.data.frame(best_auc_results, max_auc_row)
  
}

# Print the best F1 scores for each 'TrainingData'
print(best_f1_results)
print(best_precision_results)
print(best_recall_results)
print(best_auc_results)


CombinedBestPerforming <- rbind.data.frame(best_f1_results,best_precision_results,best_recall_results)


CombinedBestPerforming <- CombinedBestPerforming[,c("Precision", "Recall", "F1","AUC",
                                                    "Training Data", "N epochs", "CNN Architecture", "Threshold", "Frozen")]

CombinedBestPerforming$Precision <- round(CombinedBestPerforming$Precision,2)
CombinedBestPerforming$F1 <- round(CombinedBestPerforming$F1,2)
CombinedBestPerforming$AUC <- round(CombinedBestPerforming$AUC,2)

CombinedBestPerforming$Recall<- round(CombinedBestPerforming$Recall,2)


CombinedBestPerforming$Precision <- format(CombinedBestPerforming$Precision,nsmall = 2)
CombinedBestPerforming$F1 <- format(CombinedBestPerforming$F1,nsmall = 2)
CombinedBestPerforming$AUC <- format(CombinedBestPerforming$AUC,nsmall = 2)

CombinedBestPerforming$Threshold <- format(CombinedBestPerforming$Threshold,nsmall = 2)

CombinedBestPerforming$Recall <- format(CombinedBestPerforming$Recall,nsmall = 2)

CombinedBestPerforming$`Training Data` <- as.factor(CombinedBestPerforming$`Training Data`)
levels(CombinedBestPerforming$`Training Data` )

CombinedBestPerforming$`Training Data` <- plyr::revalue(CombinedBestPerforming$`Training Data`,
                                                        c("belizegunshots" = "Belize only",
                                                          "imagesvietnam" = "Vietnam balanced",
                                                          "imagesvietnam_belize"= "Vietnam + Belize",
                                                          "imagesvietnamunbalanced" = "Vietnam unbalanced"))

CombinedBestPerforming$`CNN Architecture` <- as.factor(CombinedBestPerforming$`CNN Architecture`)

CombinedBestPerforming$`CNN Architecture` <- plyr::revalue(CombinedBestPerforming$`CNN Architecture`,
                                                           c("alexnet"  = "AlexNet" , "vgg16"  = "VGG16"))

CombinedBestPerformingFlex <- flextable::flextable(CombinedBestPerforming)
CombinedBestPerformingFlex

#flextable::save_as_docx(CombinedBestPerformingFlex, path='Table 2.docx')

