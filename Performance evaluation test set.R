library(tidyverse)

# With balanced Vietnam data
performancetablesA <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnam_/performance_tables/'
performancetablesB <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_imagesvietnam_/performance_tables/'
# 
# # With Belize data
performancetablesC <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnam_belize_/performance_tables/'
performancetablesD <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_imagesvietnam_belize_/performance_tables/'

# With unbalanced Vietnam data
performancetablesE <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_imagesvietnamunbalanced_/performance_tables'
performancetablesF <-'/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnamunbalanced_/performance_tables'

# Find best performing model
FrozenFiles <- list.files(c(performancetablesA,performancetablesB,performancetablesC,performancetablesD,
                            performancetablesE,performancetablesF),#performancetablesC),#,performancetablesC,performancetablesD,performancetablesE),
                          full.names = T)

FrozenCombined <- FrozenFiles %>% 
  lapply(read_csv) %>%                              # Store all files in list
  bind_rows                                         # Combine data sets into one data set 


FrozenCombined <- as.data.frame(FrozenCombined)
unique_training_data <- unique(FrozenCombined$`Training Data`)

FrozenCombined$Threshold <- 1-  FrozenCombined$Threshold
performance_scores <-  FrozenCombined #subset(FrozenCombined, Threshold== 0.5  | Threshold== 0.85  |Threshold== 0.95   )

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
  max_recall_row <- subset_data[which(subset_data$Recall>.9), ]
  max_recall_row <- max_recall_row[which.max(max_recall_row$F1), ]
  best_recall_results <-rbind.data.frame(best_recall_results, max_recall_row)
}

# Print the best F1 scores for each 'TrainingData'
print(best_f1_results)
print(best_precision_results)
print(best_recall_results)


bestF1 <- performance_scores[order(performance_scores$F1,decreasing=TRUE), ]

bestF1[1:5,]


CombinedBestPerforming <- rbind.data.frame(best_f1_results,best_precision_results,best_recall_results)


CombinedBestPerforming <- CombinedBestPerforming[,c("Precision", "Recall", "F1",
                                                    "Training Data", "N epochs", "CNN Architecture", "Threshold", "Frozen")]

CombinedBestPerforming$Precision <- round(CombinedBestPerforming$Precision,2)
CombinedBestPerforming$F1 <- round(CombinedBestPerforming$F1,2)

CombinedBestPerforming$Threshold <- format(CombinedBestPerforming$Threshold,nsmall = 2)

CombinedBestPerforming$Recall <- format(CombinedBestPerforming$Recall,nsmall = 2)
CombinedBestPerforming$`Training Data` <- as.factor(CombinedBestPerforming$`Training Data`)
levels(CombinedBestPerforming$`Training Data` )

CombinedBestPerforming$`Training Data` <- plyr::revalue(CombinedBestPerforming$`Training Data`,
              c("imagesvietnam" = "Vietnam balanced",
                "imagesvietnam_belize"= "Vietnam + Belize",
                "imagesvietnamunbalanced" = "Vietnam unbalanced"))

CombinedBestPerforming$`CNN Architecture` <- as.factor(CombinedBestPerforming$`CNN Architecture`)

CombinedBestPerforming$`CNN Architecture` <- plyr::revalue(CombinedBestPerforming$`CNN Architecture`,
              c("alexNet"  = "AlexNet" ))

CombinedBestPerformingFlex <- flextable::flextable(CombinedBestPerforming)

flextable::save_as_docx(CombinedBestPerformingFlex, path='Table 2.docx')


# Re-create histogram -----------------------------------------------------
# Create a vector of hours as ranges (0-1, 2-3, etc.)
hour_ranges <- sprintf("%d-%d", seq(0, 22, by = 2), seq(1, 23, by = 2))

# Generate random counts
counts <-c(0,0,.05,.05,.1,.18,.18,.14,.08,.08,.05,0)

# Create the data frame
df <- data.frame(Dates = hour_ranges, Count = counts)

# Print the data frame
print(df)

# Create bar plot
ggbarplot(data=df,x='Dates',y='Count', fill='lightgrey')+
  xlab('Time of Day') + ylab('Number of gunshots detected \n (shots per hour)')




data.frame(Dates = c('00- 02','02-04','04-08',
                     ), Count=c(1))
