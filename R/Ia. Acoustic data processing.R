library(gibbonR)
library(tuneR)
library(seewave)
library(signal)
library(stringr)

# Create short .wav clips -------------------------------------------------

# List text files in directory
ListSelectionTables <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
           pattern = '.txt',full.names =T)

ListSelectionTablesShort <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                  pattern = '.txt',full.names =F)

# List .wav files in directory
ListWavFiles <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                  pattern = '.wav',full.names =T)

ListWavFilesShort <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                           pattern = '.wav',full.names =F)

ListWavFilesShort <- str_split_fixed(ListWavFilesShort,pattern = '.wav',n=2)[,1]

# Prepare clips
for( a in 1: length(ListSelectionTables)){
  print(paste('processing', a, 'out of',length(ListSelectionTables)))
  TempSelection <- read.delim( ListSelectionTables[a])
  TempWav <- readWave(ListWavFiles[a])
  shortSoundFile <- lapply(1:(nrow(TempSelection)),
                              function(i)
                                extractWave(
                                  TempWav,
                                  from = TempSelection$Begin.Time..s.[i]-1.5,
                                  to = TempSelection$End.Time..s.[i]+1.5,
                                  xunit = c("time"),
                                  plot = F,
                                  output = "Wave"
                                ))

  WavName <- ListWavFilesShort[a]

  lapply(1:(length(shortSoundFile)),
         function(i)
           writeWave(
             shortSoundFile[[i]],
             filename = paste('data/clips/gunshot/',WavName,i,'.wav',sep='_'),
             extensible = F
           ))

  # # Create noise clips for selections with >1 gunshots
  # if(nrow(TempSelection)>1){
  #
  #  Temp.Noise.Wavs <-  extractWave(
  #     TempWav,
  #     from = TempSelection$End.Time..s.[1],
  #     to = TempSelection$Begin.Time..s.[2],
  #     xunit = c("time"),
  #     plot = F,
  #     output = "Wave"
  #   )
  #
  #  TempSeq <- seq(1,duration(Temp.Noise.Wavs),3)
  #  length.seq <- length(TempSeq)-1
  #
  #  shortNoiseFile <- lapply(1:(length.seq),
  #                           function(i)
  #                             extractWave(
  #                               Temp.Noise.Wavs,
  #                               from = TempSeq[i],
  #                               to = TempSeq[i+1],
  #                               xunit = c("time"),
  #                               plot = F,
  #                               output = "Wave"
  #                             ))
  #
  #  lapply(1:length(shortNoiseFile),
  #         function(i)
  #           writeWave(
  #             shortNoiseFile[[i]],
  #             filename = paste('data/clips/noise/',WavName,i,'.wav',sep='_'),
  #             extensible = F
  #           ))
  # }
  #

}



# Create images for training ----------------------------------------------
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Image creation -----------------------------------------------

# The splits are set to ensure all data (100%) goes into the relevant folder.
gibbonNetR::spectrogram_images(
  trainingBasePath = 'data/clips', #'/Volumes/DJC Files/Danum Deep Learning/TestClips', #
  outputBasePath   = 'data/imagesvietnam/',
  splits           = c(0.6, 0.2, 0.2)  # 0% training, 0% validation, 100% testing
)

# Load necessary libraries
# This will be part of the package, so you may need to list these dependencies in your package DESCRIPTION file.
library(stringr)

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[,2], components[,3], sep = "_")
  return(identifier)
}

# Load necessary libraries

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
  testDir <- file.path(rootDir, 'test')
  
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

