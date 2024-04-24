# Load necessary libraries
library(stringr)  # For string manipulation
library(caret)    # For machine learning and model evaluation
library(ggpubr)   # For data visualization
library(dplyr)    # For data manipulation
library(data.table) # For sorting the detections
library(ggplot2)

# NOTE you need to change the file paths below to where your files are located on your computer

# KSWS Performance Binary --------------------------------------------------------
Annotated.files <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/',
                              full.names = T, pattern = '.txt')


# Get a list of TopModel result files
TopModelresults <- list.files('data/DetectionSelections/AlexNetbelize',full.names = T)

# Get a list of annotation selection table files
TestDataSet <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/',
                          full.names = T, pattern = '.txt')

# Preallocate space for TopModelDetectionDF
TopModelDetectionDF <- data.frame()

# Loop through each TopModel result file
for (a in 1:length(TopModelresults)) {
  
  # Read the TopModel result table into a data frame
  TempTopModelTable <- read.delim2(TopModelresults[a])
  
  # Extract the short name of the TopModel result file
  ShortName <- basename(TopModelresults[a])
  ShortName <- str_split_fixed(ShortName, pattern = '.wav', n = 2)[, 1]
  ShortName <- str_split_fixed(ShortName, pattern = 'Gunshot', n = 2)[, 1]
  
  # Find the corresponding annotation selection table
  testDataIndex <- which(str_detect(TestDataSet, ShortName))
  TestDataTable <- read.delim2(TestDataSet[testDataIndex])
  
  
  # Round Begin.Time..s. and End.Time..s. columns to numeric
  TestDataTable$Begin.Time..s. <- round(as.numeric(TestDataTable$Begin.Time..s.))
  TestDataTable$End.Time..s. <- round(as.numeric(TestDataTable$End.Time..s.))
  
  DetectionList <- list()
  # Loop through each row in TempTopModelTable
  for (c in 1:nrow(TempTopModelTable)) {
    TempRow <- TempTopModelTable[c,]
    
    # Check if Begin.Time..s. is not NA
    if (!is.na(TempRow$Begin.Time..s.)) {
      # Convert Begin.Time..s. and End.Time..s. to numeric
      TempRow$Begin.Time..s. <- as.numeric(TempRow$Begin.Time..s.)
      TempRow$End.Time..s. <- as.numeric(TempRow$End.Time..s.)
      
      # Determine if the time of the detection is within the time range of an annotation
      TimeBetween <- data.table::between(TempRow$Begin.Time..s.,
                                         TestDataTable$Begin.Time..s. - 2,
                                         TestDataTable$Begin.Time..s. + 2)
      
      # Extract the detections matching the time range
      matched_detections <- TestDataTable[TimeBetween, ]
      
      if (nrow(matched_detections) > 0) {
        # Set Class based on the Call.Type in matched_detections
        TempRow$Class <- 'Gunshot'
        DetectionList[[length( unlist(DetectionList))+1]] <-  which(TimeBetween == TRUE)
      } else {
        # Set Class to 'Noise' if no corresponding annotation is found
        TempRow$Class <- 'Noise'
      }
      
      # Append TempRow to TopModelDetectionDF
      TopModelDetectionDF <- rbind.data.frame(TopModelDetectionDF, TempRow)
    }
  }
  
  # Identify missed detections
  
  
  if (length( unlist(DetectionList)) > 0 &  length( unlist(DetectionList)) < nrow(TestDataTable) ) {
    
    missed_detections <- TestDataTable[-unlist(DetectionList), ]
    # Prepare missed detections data
    missed_detections <- missed_detections[, c("Selection", "View", "Channel", "Begin.Time..s.", "End.Time..s.", "Low.Freq..Hz.", "High.Freq..Hz.")]
    missed_detections$Probability <- 0
    missed_detections$Detections <- ShortName
    missed_detections$Class <- 'Gunshot'
    
    # Append missed detections to TopModelDetectionDF
    TopModelDetectionDF <- rbind.data.frame(TopModelDetectionDF, missed_detections)
  }
  
  if (length( unlist(DetectionList)) == 0) {
    
    missed_detections <- TestDataTable
    # Prepare missed detections data
    missed_detections <- missed_detections[, c("Selection", "View", "Channel", "Begin.Time..s.", "End.Time..s.", "Low.Freq..Hz.", "High.Freq..Hz.")]
    missed_detections$Probability <- 0
    missed_detections$Detections <- ShortName
    missed_detections$Class <- 'Gunshot'
    
    # Append missed detections to TopModelDetectionDF
    TopModelDetectionDF <- rbind.data.frame(TopModelDetectionDF, missed_detections)
    
  }
  
}

head(TopModelDetectionDF)
nrow(TopModelDetectionDF)
table(TopModelDetectionDF$Class)




# Display unique values in the Class column
unique(TopModelDetectionDF$Class)

# Define a vector of confidence Thresholds
Thresholds <-seq(0.1,1,0.1)

# Create an empty data frame to store results
BestF1data.frameGunshotBinary <- data.frame()

# Loop through each threshold value
for(a in 1:length(Thresholds)){
  
  # Filter the subset based on the confidence threshold
  TopModelDetectionDF_single <-TopModelDetectionDF
  
  TopModelDetectionDF_single$PredictedClass <-  
    ifelse(TopModelDetectionDF_single$Probability  <=Thresholds[a], 'Noise','Gunshot')
  
  # Calculate confusion matrix using caret package
  caretConf <- caret::confusionMatrix(
    as.factor(TopModelDetectionDF_single$PredictedClass),
    as.factor(TopModelDetectionDF_single$Class),positive = 'Gunshot',
    mode = 'everything')
  
  
  # Extract F1 score, Precision, and Recall from the confusion matrix
  F1 <- caretConf$byClass[7]
  Precision <- caretConf$byClass[5]
  Recall <- caretConf$byClass[6]
  FP <- caretConf$table[1,2]
  TN <- sum(caretConf$table[2,])#+JahooAdj
  FPR <-  FP / (FP + TN)
  # Create a row for the result and add it to the BestF1data.frameGreyGibbon
  #TrainingData <- training_data_type
  TempF1Row <- cbind.data.frame(F1, Precision, Recall,FPR)
  TempF1Row$Thresholds <- Thresholds[a]
  BestF1data.frameGunshotBinary <- rbind.data.frame(BestF1data.frameGunshotBinary, TempF1Row)
}

BestF1data.frameGunshotBinary

GunshotMax <- round(max(na.omit(BestF1data.frameGunshotBinary$F1)),2)

# Metric plot
GunshotBinaryPlot <- ggplot(data = BestF1data.frameGunshotBinary, aes(x = Thresholds)) +
  geom_line(aes(y = F1, color = "F1", linetype = "F1")) +
  geom_line(aes(y = Precision, color = "Precision", linetype = "Precision")) +
  geom_line(aes(y = Recall, color = "Recall", linetype = "Recall")) +
  labs(title = paste("Crested Gibbons (binary) \n max F1:",GunshotMax),
       x = "Confidence",
       y = "Values") +
  scale_color_manual(values = c("F1" = "blue", "Precision" = "red", "Recall" = "green"),
                     labels = c("F1", "Precision", "Recall")) +
  scale_linetype_manual(values = c("F1" = "dashed", "Precision" = "dotted", "Recall" = "solid")) +
  theme_minimal() +
  theme(legend.title = element_blank())+
  labs(color  = "Guide name", linetype = "Guide name", shape = "Guide name")

GunshotBinaryPlot
