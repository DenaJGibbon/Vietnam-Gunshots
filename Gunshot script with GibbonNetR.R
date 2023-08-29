library(devtools)
load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Ensure no data leakage between train, valid, and test sets --------------
# Load necessary libraries
library(stringr)

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[,2], components[,3], sep = "_")
  return(identifier)
}

# Retrieve lists of files from the respective folders
trainingDir <- 'data/imagesvietnam/train'
validationDir <- 'data/imagesvietnam/valid'
testDir <- 'data/imagesvietnam/test/'


trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)
testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = T)

# Extract identifiers for each file in the respective datasets
trainIds <- sapply(trainFiles, extract_file_identifier)
validationIds <- sapply(validationFiles, extract_file_identifier)
testIds <- sapply(testFiles, extract_file_identifier)

# Check for data leakage
trainValidationOverlap <- intersect(trainIds, validationIds)
trainTestOverlap <- intersect(trainIds, testIds)
validationTestOverlap <- intersect(validationIds, testIds)

# Report findings
if (length(trainValidationOverlap) == 0 & length(trainTestOverlap) == 0 & length(validationTestOverlap) == 0) {
  cat("No data leakage detected among the datasets.\n")
} else {
  cat("Data leakage detected!\n")
  if (length(trainValidationOverlap) > 0) {
    cat("Overlap between training and validation datasets:\n", trainValidationOverlap, "\n")
  }
  
  if (length(trainTestOverlap) > 0) {
    cat("Overlap between training and test datasets:\n", trainTestOverlap, "\n")
  }
  
  if (length(validationTestOverlap) > 0) {
    cat("Overlap between validation and test datasets:\n", validationTestOverlap, "\n")
  }
}


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam/'

# Location of spectrogram images for testing
test.data.path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam'

# Whether to unfreeze the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Location to save the out
output.data.path <-paste('data/','output','unfrozen',unfreeze.param,trainingfolder.short,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping

gibbonNetR::train_alexnet(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = FALSE,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="gunshot",
                          negative.class="noise")


gibbonNetR::train_VGG16(input.data.path=input.data.path,
                        test.data=test.data.path,
                        unfreeze = FALSE,
                        epoch.iterations=epoch.iterations,
                        early.stop = "yes",
                        output.base.path = "data/",
                        trainingfolder=trainingfolder.short,
                        positive.class="gunshot",
                        negative.class="noise")


performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnam_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir)

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
PerformanceOutput$best_auc$AUC
