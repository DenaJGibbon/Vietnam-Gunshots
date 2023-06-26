# Load required packages
library(data.table)
library(dplyr)

Annotated.files <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance',
           pattern='.txt',full.names = T)

Prediction.files <- list.files('/Users/denaclink/Desktop/RStudio Projects/Vietnam-Gunshots/data/DetectionSelections/AlexNetFrozen1epoch',
                               pattern='.txt',full.names = T)



annotation_file <- Annotated.files[[2]]
prediction_file  <- Prediction.files[[2]]

threshold <- .9
# Function to evaluate model performance

# Read annotation file
  annotations <- fread(annotation_file)

  # Read prediction file
  predictions <- fread(prediction_file)

  # Convert to data frames and rename columns
  annotations <- as.data.frame(annotations[,c(4,5)])
  colnames(annotations) <- c("start", "stop")

  predictions <- as.data.frame(predictions[,c(4,5,8)])
  colnames(predictions) <- c("start", "stop", "probability")

  # Convert start and stop columns to numeric
  annotations$start <- as.numeric(annotations$start)
  annotations$stop <- as.numeric(annotations$stop)
  predictions$start <- as.numeric(predictions$start)
  predictions$stop <- as.numeric(predictions$stop)

  # Filter predictions based on threshold
  predictions <- predictions %>% filter(probability >= threshold)

  # Initialize evaluation metrics
  true_positives <- 0
  false_positives <- 0
  false_negatives <- 0

  # Evaluate each annotation
    annotation_start <- annotations$start
    annotation_stop <- annotations$stop

    # Check if there is a prediction that overlaps with the annotation
    overlapping_predictions <- predictions %>%
      filter(start <= annotation_stop & stop >= annotation_start)

     Index <- which(predictions$start==overlapping_predictions$start)

    # Remove the matched prediction from the list
      predictions <- predictions[-Index, ]

      true_positives <-  nrow(overlapping_predictions)

      # Count remaining predictions as false positives
     false_positives <- nrow(predictions)

  # Calculate evaluation metrics
  precision <- true_positives / (true_positives + false_positives)
  recall <- true_positives / (true_positives + false_negatives)
  f1_score <- 2 * precision * recall / (precision + recall)

  # Print evaluation metrics
  cat("Evaluation Metrics:\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")



# Set the file paths for annotation and prediction
#evaluate_model(annotation_file, prediction_file, threshold = 0.5)

# File 1: F1=0.66, recall=1, precision=0.5
  # File 2: F1=0.5, recall=1, precision=0.33

  mean(c(0.66, 0.5)) # F1
  mean(c(0.5, 0.33)) # Precision

