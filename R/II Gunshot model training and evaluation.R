# A. Vietnam Binary Model Training (unbalanced) ---------------------------------------------------------
setwd("/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots")

# Load necessary packages and functions
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Number of epochs to include
epoch.iterations <- c(1, 2, 3, 4, 5, 20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.25,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}


# B. Vietnam Binary Model Training (balanced)  ---------------------------------------------------------
# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.5,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}

# C. Vietnam Binary Model Training (plus Belize) ---------------------------------------------------------

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam_belize/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam_belize'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.5,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}

