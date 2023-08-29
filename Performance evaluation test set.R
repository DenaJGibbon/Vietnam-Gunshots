library(tidyverse)

# With balanced Vietnam data
performancetablesA <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_images_balanced_/performance_tables/'
performancetablesB <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_images_balanced_/performance_tables/'
# 
# # With unbalanced Vietnam data
performancetablesC <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_images_/performance_tables'
# 
# # With added Belize data
# performancetablesD <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/output_frozenbin_trainaddedclean/performance_tables'
# performancetablesE <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/output_unfrozenbin_trainaddedclean/performance_tables'

# Find best performing model
FrozenFiles <- list.files(c(performancetablesA,performancetablesB,performancetablesC),#,performancetablesC,performancetablesD,performancetablesE),
                          full.names = T)

FrozenCombined <- FrozenFiles %>% 
  lapply(read_csv) %>%                              # Store all files in list
  bind_rows                                         # Combine data sets into one data set 


FrozenCombined <- as.data.frame(FrozenCombined)

unique_training_data <- unique(FrozenCombined$`Training Data`)

performance_scores <-FrozenCombined# subset(FrozenCombined, Threshold == .5   )

best_f1_results <- data.frame()
best_precision_results <- data.frame()
best_recall_results <- data.frame()

# Loop through each 'TrainingData' type and find the row with the maximum F1 score
for (td in unique_training_data) {
  subset_data <- subset(performance_scores, `Training Data` == td   )
  max_f1_row <- subset_data[which.max(subset_data$F1), ]
  best_f1_results <- rbind.data.frame(best_f1_results, max_f1_row)
  #max_precision_row <- subset_data[which(subset_data$Precision==1), ]
  max_precision_row <- subset_data[which.max(subset_data$Precision), ]
  best_precision_results <-rbind.data.frame(best_precision_results, max_precision_row)
  max_recall_row <- subset_data[which(subset_data$Recall ==1), ]
  max_recall_row <- max_recall_row[which.max(max_recall_row$F1), ]
  best_recall_results <-rbind.data.frame(best_recall_results, max_recall_row)
}

# Print the best F1 scores for each 'TrainingData'
print(best_f1_results)
print(best_precision_results)
print(best_recall_results)


bestF1 <- performance_scores[order(performance_scores$F1,decreasing=TRUE), ]

bestF1[1:2,]

max_recall_performance_scores <- performance_scores[which(performance_scores$Recall ==1), ]

bestRecall <- max_recall_performance_scores[order(max_recall_performance_scores$F1,decreasing=TRUE), ]
bestRecall[1:2,]


