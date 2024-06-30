library(devtools)

# Set working directory
setwd("/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots")

# Load necessary packages and functions
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[, 1], components[, 2], components[, 3], components[, 4], sep = "_")
  return(identifier)
}

# Function to check for data leakage
check_data_leakage <- function(rootDir) {
  trainingDir <- file.path(rootDir, 'train')
  validationDir <- file.path(rootDir, 'valid')
  testDir <- file.path('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/testdatacombined/test/')
  
  trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  
  trainIds <- sapply(trainFiles, extract_file_identifier)
  validationIds <- sapply(validationFiles, extract_file_identifier)
  testIds <- sapply(testFiles, extract_file_identifier)
  
  trainValidationOverlap <- trainIds[which(trainIds %in% validationIds)]
  trainTestOverlap <- trainIds[which(trainIds %in% testIds)]
  validationTestOverlap <- testIds[which(testIds %in% validationIds)]
  
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

# Check for leakage in different datasets
check_data_leakage('data/imagesvietnam')
check_data_leakage('data/imagesvietnam_belize')

# Check for data leakage in final test .wavs
trainingDir <- file.path('data/imagesvietnam/', 'train')
trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
trainIds <- sapply(trainFiles, extract_file_identifier)
FinalTestWavs <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance', pattern = 'wav')
FinalTestWavs <- str_split_fixed(FinalTestWavs, pattern = '.wav', n = 2)[, 1]

# No data leakage here; If true will return index of files with leakage
which(str_detect(trainIds, FinalTestWavs[1]))
which(str_detect(trainIds, FinalTestWavs[2]))

# Define paths for data
input_data_path <- 'data/imagesvietnamunbalanced/'
test_data_path <- 'data/testdatacombined/'
training_folder_short <- 'imagesvietnamunbalanced'

# Set parameters
unfreeze_param <- FALSE
epoch_iterations <- c(1, 2, 3, 4, 5, 20)
output_data_path <- paste('data/', 'output', 'unfrozen', unfreeze_param, training_folder_short, '/', sep = '_')

# Create output directory if it doesn't exist
dir.create(output_data_path)

# Train models
gibbonNetR::train_alexNet(
  input.data.path = input_data_path,
  test.data = test_data_path,
  unfreeze = unfreeze_param,
  epoch.iterations = epoch_iterations,
  early.stop = "yes",
  output.base.path = "data/",
  trainingfolder = training_folder_short,
  positive.class = "gunshot",
  negative.class = "noise"
)

gibbonNetR::train_VGG16(
  input.data.path = input_data_path,
  test.data = test_data_path,
  unfreeze = unfreeze_param,
  epoch.iterations = epoch_iterations,
  early.stop = "yes",
  output.base.path = "data/",
  trainingfolder = training_folder_short,
  positive.class = "gunshot",
  negative.class = "noise"
)

# Analyze performance
performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnam_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir = performancetables.dir, class='gunshot')

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
as.data.frame(PerformanceOutput$best_f1)
as.data.frame(PerformanceOutput$best_precision)

# Analyze performance
performancetables.dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/_output_unfrozen_FALSE_imagesvietnam_belize_/performance_tables/'
PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir = performancetables.dir, 
                                                      model.type='NA')

PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
as.data.frame(PerformanceOutput$best_f1)
as.data.frame(PerformanceOutput$best_precision)

