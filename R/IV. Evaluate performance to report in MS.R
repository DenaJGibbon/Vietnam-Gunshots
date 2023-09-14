  # Load required packages
  library(data.table)
  library(dplyr)
  
  
  Annotated.files <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/',
                                full.names = T, pattern = '.txt')
 
  ListDirs <- list.files('data/DetectionSelections',full.names = T)
  
  PerformanceDF <- data.frame()
  threshold <- 0.9
  for(k in 1:length(ListDirs)){
    
    Prediction.files <- list.files(ListDirs[k],
                                  pattern='.txt',full.names = T)
    
    Prediction.files.short <- basename(Annotated.files)
  
    for( j in 1:length(Annotated.files)){
    # Read files
    annotation_file <- Annotated.files[[j]]  # Replace with actual index
    prediction_file  <- Prediction.files[[j]]  # Replace with actual index
    

    
    # Read annotation and prediction files
    annotations <- fread(annotation_file)
    predictions <- fread(prediction_file)
    
    # Rename and filter columns
    annotations <- as.data.frame(annotations[,c(4,5)])
    colnames(annotations) <- c("start", "stop")
    
    predictions <- as.data.frame(predictions[,c(4,5,8)])
    colnames(predictions) <- c("start", "stop", "probability")
    
    # Filter based on threshold
    predictions <- dplyr::filter(predictions, probability >= threshold)
    
    # Initialize counters
    true_positives <- 0
    false_positives <- 0
    false_negatives <- 0
    
    # Loop through each annotation
    for (i in 1:nrow(annotations)) {
      annotation_start <- annotations$start[i]
      annotation_stop <- annotations$stop[i]
      
      # Check for overlapping predictions
      overlapping_predictions <- dplyr::filter(predictions, start <= annotation_stop & stop >= annotation_start)
      
      if (nrow(overlapping_predictions) > 0) {
        true_positives <- true_positives + 1
        index <- which(predictions$start == overlapping_predictions$start[1])
        predictions <- predictions[-index,]
      } else {
        false_negatives <- false_negatives + 1
      }
    }
    
    # Remaining predictions are false positives
    false_positives <- nrow(predictions)
    
    
    # Calculate metrics
    precision <- true_positives / (true_positives + false_positives)
    recall <- true_positives / (true_positives + false_negatives)
    if(is.na(precision) == FALSE ){
    f1_score <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
    } else {
      f1_score <- 0
    }
    short_name <- Prediction.files.short[j]
    
    TempPerformance <-cbind.data.frame(precision,recall,f1_score,short_name)
    TempPerformance$TrainingData <- basename(ListDirs[k])
    PerformanceDF <- rbind.data.frame(PerformanceDF,TempPerformance)
    # Print metrics
    cat("Evaluation Metrics:\n")
    cat("Precision:", precision, "\n")
    cat("Recall:", recall, "\n")
    cat("F1 Score:", f1_score, "\n")
    
  }
  }
  
  mean(PerformanceDF$recall)
  mean(PerformanceDF$precision)
  mean(PerformanceDF$f1_score) 
  
  
  
  PerformanceDF %>%
    group_by(TrainingData) %>%
    summarise_at(vars(recall,precision,f1_score), list(mean = mean))
  