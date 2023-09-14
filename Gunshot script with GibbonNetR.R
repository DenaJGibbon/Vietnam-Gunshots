library(devtools)
setwd("/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots")

# Create images for training ----------------------------------------------
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Image creation -----------------------------------------------

# The splits are set to ensure data goes into the relevant folder.
# gibbonNetR::spectrogram_images(
#   trainingBasePath = 'data/clips', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
#   outputBasePath   = 'data/imagesvietnamunbalanced/',
#   splits           = c(0.6, 0.2, 0.2)  # 0% training, 0% validation, 100% testing
# )


# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[,1],components[,2], components[,3], components[,4], sep = "_")
  return(identifier)
}

# Main function to check for data leakage
check_data_leakage <- function(rootDir) {
  # Construct paths to the subdirectories
  trainingDir <- file.path(rootDir, 'train')
  validationDir <- file.path(rootDir, 'valid')
  testDir <- file.path('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/testdatacombined/test/')
  
  # Retrieve lists of files from the respective folders
  trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  
  # Extract identifiers for each file in the respective datasets
  trainIds <- sapply(trainFiles, extract_file_identifier)
  validationIds <- sapply(validationFiles, extract_file_identifier)
  testIds <- sapply(testFiles, extract_file_identifier)
  
  # Check for data leakage
  trainValidationOverlap <- trainIds[which(trainIds %in% validationIds)]
  trainTestOverlap <- trainIds[which(trainIds %in% testIds)]
  validationTestOverlap <- testIds[which(testIds %in% validationIds)]
  
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
}

# Check for leakage
check_data_leakage('data/imagesvietnam')
check_data_leakage('data/imagesvietnam_belize')


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test.data.path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Whether to unfreeze the layers
unfreeze.param <- FALSE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Location to save the out
output.data.path <-paste('data/','output','unfrozen',unfreeze.param,trainingfolder.short,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping

gibbonNetR::train_alexNet(input.data.path=input.data.path,
                          test.data=test.data.path,
                          unfreeze = unfreeze.param,
                          epoch.iterations=epoch.iterations,
                          early.stop = "yes",
                          output.base.path = "data/",
                          trainingfolder=trainingfolder.short,
                          positive.class="gunshot",
                          negative.class="noise")


gibbonNetR::train_VGG16(input.data.path=input.data.path,
                        test.data=test.data.path,
                        unfreeze = unfreeze.param,
                        epoch.iterations=epoch.iterations,
                        early.stop = "yes",
                        output.base.path = "data/",
                        trainingfolder=trainingfolder.short,
                        positive.class="gunshot",
                        negative.class="noise")


performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_TRUE_imagesvietnam_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                      'gunshot')

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
as.data.frame(PerformanceOutput$best_f1)
as.data.frame(PerformanceOutput$best_precision)
