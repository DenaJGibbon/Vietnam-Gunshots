library(stringr)

# Part 1. Check for leakage -----------------------------------------------

# Function to extract the relevant identifier from the filename
extract_file_identifier <- function(filename) {
  components <- str_split_fixed(filename, "_", n = 5)
  identifier <- paste(components[, 1], components[, 2], components[, 3], components[, 4], sep = "_")
  return(identifier)
}

rootDir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/imagesvietnamunbalanced/'

# Function to check for data leakage
check_data_leakage <- function(rootDir) {
  trainingDir <- file.path(rootDir, 'train/')
  validationDir <- file.path(rootDir, 'valid/')
  testDir <- file.path('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/testdatacombined/test/')
  #testDir <-file.path(rootDir, 'test')
  
  trainFiles <- list.files(trainingDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  validationFiles <- list.files(validationDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  testFiles <- list.files(testDir, pattern = "\\.jpg$", full.names = FALSE, recursive = TRUE)
  
  trainIds <- sapply(trainFiles, extract_file_identifier)
  validationIds <- sapply(validationFiles, extract_file_identifier)
  testIds <- sapply(testFiles, extract_file_identifier)
  
  trainValidationOverlap <- trainIds[which(trainIds %in% validationIds)]
  trainTestOverlap <- trainIds[which(trainIds %in% testIds)]
  validationTestOverlap <- testIds[which(testIds %in% validationIds)]
  
  if(length(trainFiles)==0|length(validationFiles)==0 | length(testFiles) ==0){
    print('Zero files in train, valid, or test path cannot check for leakage')
    break
  }
  
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
check_data_leakage('data/imagesvietnamunbalanced/')


# Part 2a. Create matching folder with clips for future benchmarking -------

ImagesFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/imagesvietnamunbalanced/',
                         recursive = T,full.names = T)

ImagesFileShort <- basename(ImagesFile)
ImagesFileShort <- str_split_fixed(ImagesFileShort,pattern = '.jpg', n=2)[,1]

ImagesFileFolders <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/imagesvietnamunbalanced/',
                         recursive = T,full.names = F)

TrainFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,1]
CategoryFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,2]


WavFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/clips/',
                      recursive = T,full.names = T)

WavFileShort <- basename(WavFile)
WavFileShort <- str_split_fixed(WavFileShort,pattern = '.wav', n=2)[,1]

OutputDir <- '/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/clipsforzenodo/imagesvietnamunbalanced_clips/'

for(a in 1:length(ImagesFileShort)){

  # Extract wav file path
FullWavPath <- WavFile[which((WavFileShort %in%  ImagesFileShort[a]))]
ShortWav <-  WavFileShort[which((WavFileShort %in%  ImagesFileShort[a]))]
CombinedOutputDir <- paste(OutputDir,TrainFolder[a],CategoryFolder[a],sep='/')

dir.create(CombinedOutputDir,recursive = T)  

print(CombinedOutputDir)

file.copy(
  from=FullWavPath,
  to= paste(CombinedOutputDir, '/', ShortWav,'.wav',sep=''))

}


# Part 2b. Create matching folder with clips and Belize data for f --------

ImagesFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/imagesvietnam_belize/',
                         recursive = T,full.names = T)

ImagesFileShort <- basename(ImagesFile)
ImagesFileShort <- str_split_fixed(ImagesFileShort,pattern = '.jpg', n=2)[,1]
ImagesFileShort <- str_split_fixed(ImagesFileShort,pattern = '.WAV', n=2)[,1]

ImagesFileFolders <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/imagesvietnam_belize/',
                                recursive = T,full.names = F)

TrainFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,1]
CategoryFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,2]


WavFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/clips/',
                      recursive = T,full.names = T)
WavFile1 <- list.files('/Volumes/DJC Files/GunshotDataWavBelize/Training data reduced/',
                       recursive = T,full.names = T)

WavFile <- c(WavFile,WavFile1)
WavFileShort <- basename(WavFile)
WavFileShort <- str_split_fixed(WavFileShort,pattern = '.wav', n=2)[,1]
WavFileShort <- str_split_fixed(WavFileShort,pattern = '.WAV', n=2)[,1]

OutputDir <- '/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/clipsforzenodo/imagesvietnam_belize_clips/'

for(a in 1:length(ImagesFileShort)){
  
  # Extract wav file path
  FullWavPath <- WavFile[which((WavFileShort %in%  ImagesFileShort[a]))]
  ShortWav <-  WavFileShort[which((WavFileShort %in%  ImagesFileShort[a]))]
  CombinedOutputDir <- paste(OutputDir,TrainFolder[a],CategoryFolder[a],sep='/')
  
  dir.create(CombinedOutputDir,recursive = T)  
  
  print(CombinedOutputDir)
  
  file.copy(
    from=FullWavPath,
    to= paste(CombinedOutputDir, '/', ShortWav,'.wav',sep=''))
  
}

# Part 2c. Create matching folder with clips for future benchmarking -------

ImagesFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/testdatacombined/',
                         recursive = T,full.names = T)

ImagesFileShort <- basename(ImagesFile)
ImagesFileShort <- str_split_fixed(ImagesFileShort,pattern = '.jpg', n=2)[,1]

ImagesFileFolders <- list.files('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/testdatacombined/',
                                recursive = T,full.names = F)

TrainFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,1]
CategoryFolder <- str_split_fixed(ImagesFileFolders,pattern = '/',n=3)[,2]


WavFile <- list.files('/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/data/clips/',
                      recursive = T,full.names = T)

WavFileShort <- basename(WavFile)
WavFileShort <- str_split_fixed(WavFileShort,pattern = '.wav', n=2)[,1]

OutputDir <- '/Users/denaclink/Desktop/RStudioProjects/CleanGunshot/clipsforzenodo/testdatacombined_clips/'

for(a in 1:length(ImagesFileShort)){
  
  # Extract wav file path
  FullWavPath <- WavFile[which((WavFileShort %in%  ImagesFileShort[a]))]
  ShortWav <-  WavFileShort[which((WavFileShort %in%  ImagesFileShort[a]))]
  CombinedOutputDir <- paste(OutputDir,TrainFolder[a],CategoryFolder[a],sep='/')
  
  dir.create(CombinedOutputDir,recursive = T)  
  
  print(CombinedOutputDir)
  
  file.copy(
    from=FullWavPath,
    to= paste(CombinedOutputDir, '/', ShortWav,'.wav',sep=''))
  
}
