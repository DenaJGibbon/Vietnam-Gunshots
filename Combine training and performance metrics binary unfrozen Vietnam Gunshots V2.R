# Combine training and testing into one script
# Packages ----------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)
library(stringr)
library(ROCR)
library(dplyr)
library(tibble)
library(readr)


# Datasets ----------------------------------------------------------------
# Note: Need to create temp folders in project  to save images
device <- if(cuda_is_available()) "cuda" else "cpu"

to_device <- function(x, device) {
  x$to(device = device)
}

# Location of spectrogram images for training 
input.data <-  c('/Users/denaclink/Desktop/RStudioProjects/Multi-species-detector/data/imagesmalaysiaHQ/')

# Location of spectrogram images for testing
test.data <- c('/Users/denaclink/Desktop/RStudioProjects/Multi-species-detector/data/imagesmalaysiamaliau/')

# Training data folder short
trainingfolder <- 'imagesmalaysiaHQ'

# Whether to unfreeze the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Location to save the out
output.data.path <-paste('/Users/denaclink/Desktop/RStudioProjects/Multi-species-detector/data/','output','unfrozen',unfreeze.param,trainingfolder,'/', sep='_')

# Create if doesn't exist
dir.create(output.data.path)

# Allow early stopping?
early.stop <- 'yes' # NOTE: Must comment out if don't want early stopping                          

# Create metadata
metadata <- tibble(
  Model_Name = "AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet152", # Modify as per your model's name
  Training_Data_Path = input.data,
  Test_Data_Path = test.data,
  Output_Path = output.data.path,
  Device_Used = device,
  EarlyStop=early.stop,
  Layers_Ununfrozen = unfreeze.param,
  Epochs = epoch.iterations
)

# Save metadata to CSV
write_csv(metadata, paste0(output.data.path, "model_metadata.csv"))


for(b in 1:length(epoch.iterations)){
  
  n.epoch <- epoch.iterations[b]
  
  # Combined uses both
  train_ds <- image_folder_dataset(
    file.path(input.data,'train' ),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_color_jitter() %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1)
  
  valid_ds <- image_folder_dataset(
    file.path(input.data, "valid"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1)
  
  
  train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)
  
  n.classes <- length(train_ds$class_to_idx)
  classes <- train_ds$classes
  
  # Train AlexNet -----------------------------------------------------------
  
  net <- torch::nn_module(
    
    initialize = function() {
      self$model <- model_alexnet(pretrained = TRUE)
      
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param) # FALSE means the features are frozen; TRUE unfrozen
      }
      
      self$model$classifier <- nn_sequential(
        nn_dropout(0.5),
        nn_linear(9216, 512),
        nn_relu(),
        nn_linear(512, 256),
        nn_relu(),
        nn_linear(256, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
    
  )
  
  
  
  fitted <- net %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  
  modelAlexNetGibbon <- fitted %>%
    fit(train_dl,epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"),
          #luz_callback_model_checkpoint(path = "cpt_AlexNet/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_AlexNet.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  luz_save(modelAlexNetGibbon, paste( output.data.path,trainingfolder,n.epoch, "modelAlexNet.pt",sep='_'))
  #modelAlexNetGibbon <- luz_load( paste( output.data.path,trainingfolder,n.epoch, "modelAlexNet.pt",sep='_'))
  
  TempCSV.AlexNet <-  read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_AlexNet.csv",sep='_'))
  
  AlexNet.loss <- TempCSV.AlexNet[nrow(TempCSV.AlexNet),]$loss
  
  
  # Train VGG16 -------------------------------------------------------------
  
  net <- torch::nn_module(
    initialize = function() {
      self$model <- model_vgg16 (pretrained = TRUE)
      
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param)
      }
      
      self$model$classifier <- nn_sequential(
        nn_dropout(0.5),
        nn_linear(25088, 4096),
        nn_relu(),
        nn_dropout(0.5),
        nn_linear(4096, 4096),
        nn_relu(),
        nn_linear(4096, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
  )
  
  fitted <- net  %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  modelVGG16Gibbon <- fitted %>%
    fit(train_dl,epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"
          ),
          #luz_callback_model_checkpoint(path = "cpt_VGG16/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_VGG16.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  luz_save(modelVGG16Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelVGG16.pt",sep='_'))
  
  
  TempCSV.VGG16 <- read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_VGG16.csv",sep='_'))
  
  VGG16.loss <- TempCSV.VGG16[nrow(TempCSV.VGG16),]$loss
  
  # Train VGG19 -------------------------------------------------------------
  
  net <- torch::nn_module(
    initialize = function() {
      self$model <- model_vgg19 (pretrained = TRUE)
      
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param)
      }
      
      self$model$classifier <- nn_sequential(
        nn_dropout(0.5),
        nn_linear(25088, 4096),
        nn_relu(),
        nn_dropout(0.5),
        nn_linear(4096, 4096),
        nn_relu(),
        nn_linear(4096, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
  )
  
  fitted <- net  %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  modelVGG19Gibbon <- fitted %>%
    fit(train_dl,epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"
          ),
          #luz_callback_model_checkpoint(path = "cpt_VGG19/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_VGG19.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  luz_save(modelVGG19Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelVGG19.pt",sep='_'))
  
  
  TempCSV.VGG19 <- read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_VGG19.csv",sep='_'))
  
  VGG19.loss <- TempCSV.VGG19[nrow(TempCSV.VGG19),]$loss
  
  # Calculate performance metrics -------------------------------------------
  dir.create(paste(output.data.path,'performance_tables',sep=''))
  
  # Get the list of image files
  imageFiles <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = TRUE)
  
  # Get the list of image files
  imageFileShort <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = FALSE)
  
  Folder <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,1]
  
  imageFileShort <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,2]
  #imageFileShort <- paste( Folder,imageFileShort,sep='')
  
  # Prepare output tables
  outputTableAlexNet <- data.frame()
  outputTableVGG16 <- data.frame()
  outputTableVGG19 <- data.frame()
  
  # Iterate over image files
  
  AlexNetProbdf <- data.frame()
  VGG16Probdf <- data.frame()
  VGG19Probdf <- data.frame()
  
  
  test_ds <- image_folder_dataset(
    file.path(test.data, "test/"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
    target_transform = function(x) as.double(x) - 1
  )
  
  # Predict the test files
  # Variable indicating the number of files
  #nfiles <- test_ds$.length()
  
  # Load the test images
  test_dl <- dataloader(test_ds, batch_size = 32, shuffle = F)
  
  # Predict using AlexNet
  AlexNetPred <- predict(modelAlexNetGibbon, test_dl)
  AlexNetProb <- torch_sigmoid(AlexNetPred)
  AlexNetProb <- as_array(torch_tensor(AlexNetProb, device = 'cpu'))
  AlexNetClass <- ifelse((AlexNetProb) < 0.5, "Gibbons", "Noise")
  
  # Predict using VGG16
  VGG16Pred <- predict(modelVGG16Gibbon, test_dl)
  VGG16Prob <- torch_sigmoid(VGG16Pred)
  VGG16Prob <- as_array(torch_tensor(VGG16Prob, device = 'cpu'))
  VGG16Class <- ifelse((VGG16Prob) < 0.5, "Gibbons", "Noise")
  
  # Predict using VGG19
  VGG19Pred <- predict(modelVGG19Gibbon, test_dl)
  VGG19Prob <- torch_sigmoid(VGG19Pred)
  VGG19Prob <- as_array(torch_tensor(VGG19Prob, device = 'cpu'))
  VGG19Class <- ifelse((VGG19Prob) < 0.5, "Gibbons", "Noise")
  
  # Add the results to output tables
  outputTableAlexNet <- rbind(outputTableAlexNet, data.frame(Label = Folder, Probability = AlexNetProb, PredictedClass = AlexNetClass, ActualClass = Folder))
  outputTableVGG16 <- rbind(outputTableVGG16, data.frame(Label = Folder, Probability = VGG16Prob, PredictedClass = VGG16Class, ActualClass = Folder))
  outputTableVGG19 <- rbind(outputTableVGG19, data.frame(Label = Folder, Probability = VGG19Prob, PredictedClass = VGG19Class, ActualClass = Folder))
  
  
  # Save the output tables as CSV files
  write.csv(outputTableAlexNet, paste(output.data.path, trainingfolder, n.epoch, "output_AlexNet.csv", sep = '_'), row.names = FALSE)
  write.csv(outputTableVGG16, paste(output.data.path, trainingfolder, n.epoch, "output_VGG16.csv", sep = '_'), row.names = FALSE)
  write.csv(outputTableVGG19, paste(output.data.path, trainingfolder, n.epoch, "output_VGG19.csv", sep = '_'), row.names = FALSE)
  
  
  # Initialize data frames
  CombinedTempRow <- data.frame()
  TransferLearningCNNDF <- data.frame()
  # Threshold values to consider
  thresholds <- seq(0.1,1,0.1)
  
  for (threshold in thresholds) {
    # AlexNet
    AlexNetPredictedClass <- ifelse((outputTableAlexNet$Probability) < threshold, "Gibbons", "Noise")
    
    AlexNetPerf <- caret::confusionMatrix(
      as.factor(AlexNetPredictedClass),
      as.factor(outputTableAlexNet$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowAlexNet <- cbind.data.frame(
      t(AlexNetPerf[5:7]),
      AlexNet.loss,
      trainingfolder,
      n.epoch,
      'AlexNet'
    )
    
    colnames(TempRowAlexNet) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowAlexNet$Threshold <- as.character(threshold)
    
    # VGG16
    VGG16PredictedClass <- ifelse((outputTableVGG16$Probability) < threshold, "Gibbons", "Noise")
    
    VGG16Perf <- caret::confusionMatrix(
      as.factor(VGG16PredictedClass),
      as.factor(outputTableVGG16$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowVGG16 <- cbind.data.frame(
      t(VGG16Perf[5:7]),
      VGG16.loss,
      trainingfolder,
      n.epoch,
      'VGG16'
    )
    
    colnames(TempRowVGG16) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowVGG16$Threshold <- as.character(threshold)
    
    # VGG19
    VGG19PredictedClass <- ifelse((outputTableVGG19$Probability) < threshold, "Gibbons", "Noise")
    
    VGG19Perf <- caret::confusionMatrix(
      as.factor(VGG19PredictedClass),
      as.factor(outputTableVGG19$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowVGG19 <- cbind.data.frame(
      t(VGG19Perf[5:7]),
      VGG19.loss,
      trainingfolder,
      n.epoch,
      'VGG19'
    )
    
    colnames(TempRowVGG19) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowVGG19$Threshold <- as.character(threshold)
    
    CombinedTempRowThreshold <- rbind.data.frame(TempRowAlexNet, TempRowVGG16, TempRowVGG19)
    CombinedTempRowThreshold$Threshold <- as.character(threshold)
    
    # Append to the overall result data frame
    CombinedTempRow <- rbind.data.frame(CombinedTempRow, CombinedTempRowThreshold)
  }
  
  # Append to the main data frame
  TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
  TransferLearningCNNDF$Frozen <- unfreeze.param
  # Write the result to a CSV file
  filename <- paste(output.data.path,'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFAlexNETVGG16.csv', sep = '')
  write.csv(TransferLearningCNNDF, filename, row.names = FALSE)
  
  rm(modelAlexNetGibbon)
  rm(modelVGG16Gibbon)
  rm(modelVGG19Gibbon)
  
  
  # Start ResNet ------------------------------------------------------------
  
  TransferLearningCNNDF <- data.frame()
  # Combined uses both
  train_ds <- image_folder_dataset(
    file.path(input.data,'train' ),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_color_jitter() %>%
      transform_resize(256) %>%
      transform_center_crop(224) %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1 )
  
  valid_ds <- image_folder_dataset(
    file.path(input.data, "valid"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      transform_resize(256) %>%
      transform_center_crop(224) %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)), target_transform = function(x) as.double(x) - 1 )
  
  
  train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)
  
  
  # Train ResNet18 -----------------------------------------------------------
  
  
  convnet <- nn_module(
    initialize = function() {
      self$model <- model_resnet18 (pretrained = TRUE)
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param) # False means the features are unfrozen
      }
      self$model$fc <- nn_sequential(
        nn_linear(self$model$fc$in_features, 1024),
        nn_relu(),
        nn_linear(1024, 1024),
        nn_relu(),
        nn_linear(1024, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
  )
  
  model <- convnet %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  # rates_and_losses <- model %>% lr_finder(train_dl)
  # rates_and_losses %>% plot()
  
  fitted <- model %>%
    fit(train_dl, epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"),
          #luz_callback_model_checkpoint(path = "output_unfrozenbin_trainaddedclean/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch,"logs_ResNet18.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  modelResNet18Gibbon <- fitted
  
  # Save model output
  luz_save(modelResNet18Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelResNet18.pt",sep='_'))
  #modelResNet18Gibbon <- luz_load("modelResNet18Gibbon1epochs.pt")
  
  TempCSV.ResNet18 <-  read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet18.csv",sep='_'))
  
  ResNet18.loss <- TempCSV.ResNet18[nrow(TempCSV.ResNet18),]$loss
  
  
  # Train ResNet50 -------------------------------------------------------------
  
  convnet <- nn_module(
    initialize = function() {
      self$model <- model_resnet50 (pretrained = TRUE)
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param) # False means the features are unfrozen
      }
      self$model$fc <- nn_sequential(
        nn_linear(self$model$fc$in_features, 1024),
        nn_relu(),
        nn_linear(1024, 1024),
        nn_relu(),
        nn_linear(1024, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
  )
  
  model <- convnet %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  # rates_and_losses <- model %>% lr_finder(train_dl)
  # rates_and_losses %>% plot()
  
  fitted <- model %>%
    fit(train_dl, epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"),
          #luz_callback_model_checkpoint(path = "output_unfrozenbin_trainaddedclean/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet50.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  modelResNet50Gibbon <- fitted
  
  # Save model output
  luz_save(modelResNet50Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelResNet50.pt",sep='_'))
  #modelResNet50Gibbon <- luz_load("modelResNet50Gibbon1epochs.pt")
  
  TempCSV.ResNet50 <-  read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet50.csv",sep='_'))
  
  ResNet50.loss <- TempCSV.ResNet50[nrow(TempCSV.ResNet50),]$loss
  # Train ResNet152 -------------------------------------------------------------
  
  convnet <- nn_module(
    initialize = function() {
      self$model <- model_resnet152 (pretrained = TRUE)
      for (par in self$parameters) {
        par$requires_grad_(unfreeze.param) # False means the features are unfrozen
      }
      self$model$fc <- nn_sequential(
        nn_linear(self$model$fc$in_features, 1024),
        nn_relu(),
        nn_linear(1024, 1024),
        nn_relu(),
        nn_linear(1024, 1)
      )
    },
    forward = function(x) {
      output <- self$model(x)
      torch_squeeze(output, dim=2)
    }
  )
  
  model <- convnet %>%
    setup(
      loss = nn_bce_with_logits_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    )
  
  
  # rates_and_losses <- model %>% lr_finder(train_dl)
  # rates_and_losses %>% plot()
  
  fitted <- model %>%
    fit(train_dl, epochs=n.epoch, valid_data = valid_dl,
        callbacks = list(
          luz_callback_early_stopping(patience = 2),
          luz_callback_lr_scheduler(
            lr_one_cycle,
            max_lr = 0.01,
            epochs=n.epoch,
            steps_per_epoch = length(train_dl),
            call_on = "on_batch_end"),
          #luz_callback_model_checkpoint(path = "output_unfrozenbin_trainaddedclean/"),
          luz_callback_csv_logger(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet152.csv",sep='_'))
        ),
        verbose = TRUE)
  
  # Save model output
  modelResNet152Gibbon <- fitted
  
  # Save model output
  luz_save(modelResNet152Gibbon, paste( output.data.path,trainingfolder,n.epoch, "modelResNet152.pt",sep='_'))
  #modelResNet152Gibbon <- luz_load("modelResNet152Gibbon1epochs.pt")
  
  TempCSV.ResNet152 <-  read.csv(paste( output.data.path,trainingfolder,n.epoch, "logs_ResNet152.csv",sep='_'))
  
  ResNet152.loss <- TempCSV.ResNet152[nrow(TempCSV.ResNet152),]$loss
  
  # Calculate performance metrics -------------------------------------------
  
  # Get the list of image files
  imageFiles <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = TRUE)
  
  # Get the list of image files
  imageFileShort <- list.files(paste(test.data,'/','test',sep=''), recursive = TRUE, full.names = FALSE)
  
  Folder <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,1]
  
  imageFileShort <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,2]
  
  # Prepare output tables
  outputTableResNet18 <- data.frame()
  outputTableResNet50 <- data.frame()
  outputTableResNet152 <- data.frame()
  
  ResNet18Probdf <- data.frame()
  ResNet50Probdf <- data.frame()
  ResNet152Probdf <- data.frame()
  
  
  test_ds <- image_folder_dataset(
    file.path(test.data, "test/"),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      transform_resize(256) %>%
      transform_center_crop(224) %>%
      transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
    target_transform = function(x) as.double(x) - 1
  )
  
  # Predict the test files
  # Variable indicating the number of files
  #nfiles <- test_ds$.length()
  
  # Load the test images
  test_dl <- dataloader(test_ds, batch_size = 32, shuffle = F)
  
  
  # Predict using ResNet18
  ResNet18Pred <- predict(modelResNet18Gibbon, test_dl)
  ResNet18Prob <- torch_sigmoid(ResNet18Pred)
  ResNet18Prob <- as_array(torch_tensor(ResNet18Prob, device = 'cpu'))
  ResNet18Class <- ifelse((ResNet18Prob) < 0.5, "Gibbons", "Noise")
  
  # Predict using ResNet50
  ResNet50Pred <- predict(modelResNet50Gibbon, test_dl)
  ResNet50Prob <- torch_sigmoid(ResNet50Pred)
  ResNet50Prob <- as_array(torch_tensor(ResNet50Prob, device = 'cpu'))
  ResNet50Class <- ifelse((ResNet50Prob) < 0.5, "Gibbons", "Noise")
  
  # Predict using ResNet152
  ResNet152Pred <- predict(modelResNet152Gibbon, test_dl)
  ResNet152Prob <- torch_sigmoid(ResNet152Pred)
  ResNet152Prob <- as_array(torch_tensor(ResNet152Prob, device = 'cpu'))
  ResNet152Class <- ifelse((ResNet152Prob) < 0.5, "Gibbons", "Noise")
  
  # Add the results to output tables
  outputTableResNet18 <- rbind(outputTableResNet18, data.frame(Label = Folder, Probability = ResNet18Prob, PredictedClass = ResNet18Class, ActualClass = Folder))
  outputTableResNet50 <- rbind(outputTableResNet50, data.frame(Label = Folder, Probability = ResNet50Prob, PredictedClass = ResNet50Class, ActualClass = Folder))
  outputTableResNet152 <- rbind(outputTableResNet152, data.frame(Label = Folder, Probability = ResNet152Prob, PredictedClass = ResNet152Class, ActualClass = Folder))
  
  # Save false positives to train next iteration
  # file.copy(Folder[which(ResNet18Prob > 0.9)],
  #           to = paste('/Users/denaclink/Desktop/RStudioProjects/Multi-species-detector/data/Temp/Images/Images/', TempShort[which(ResNet18Prob > 0.9)], sep = ''))
  
  
  # Save the output tables as CSV files
  write.csv(outputTableResNet18, paste(output.data.path, trainingfolder, n.epoch, "output_ResNet18.csv", sep = '_'), row.names = FALSE)
  write.csv(outputTableResNet50, paste(output.data.path, trainingfolder, n.epoch, "output_ResNet50.csv", sep = '_'), row.names = FALSE)
  write.csv(outputTableResNet152, paste(output.data.path, trainingfolder, n.epoch, "output_ResNet152.csv", sep = '_'), row.names = FALSE)
  
  
  
  # Initialize data frames
  CombinedTempRow <- data.frame()
  TransferLearningCNNDF <- data.frame()
  
  # Threshold values to consider
  thresholds <- seq(0.1,1,0.1)
  
  for (threshold in thresholds) {
    # ResNet18
    ResNet18PredictedClass <- ifelse((outputTableResNet18$Probability) < threshold, "Gibbons", "Noise")
    
    ResNet18Perf <- caret::confusionMatrix(
      as.factor(ResNet18PredictedClass),
      as.factor(outputTableResNet18$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowResNet18 <- cbind.data.frame(
      t(ResNet18Perf[5:7]),
      ResNet18.loss,
      trainingfolder,
      n.epoch,
      'ResNet18'
    )
    
    colnames(TempRowResNet18) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowResNet18$Threshold <- as.character(threshold)
    
    # ResNet50
    ResNet50PredictedClass <- ifelse((outputTableResNet50$Probability) < threshold, "Gibbons", "Noise")
    
    ResNet50Perf <- caret::confusionMatrix(
      as.factor(ResNet50PredictedClass),
      as.factor(outputTableResNet50$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowResNet50 <- cbind.data.frame(
      t(ResNet50Perf[5:7]),
      ResNet50.loss,
      trainingfolder,
      n.epoch,
      'ResNet50'
    )
    
    colnames(TempRowResNet50) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowResNet50$Threshold <- as.character(threshold)
    
    # ResNet152
    ResNet152PredictedClass <- ifelse((outputTableResNet152$Probability) < threshold, "Gibbons", "Noise")
    
    ResNet152Perf <- caret::confusionMatrix(
      as.factor(ResNet152PredictedClass),
      as.factor(outputTableResNet152$ActualClass),
      mode = 'everything'
    )$byClass
    
    TempRowResNet152 <- cbind.data.frame(
      t(ResNet152Perf[5:7]),
      ResNet152.loss,
      trainingfolder,
      n.epoch,
      'ResNet152'
    )
    
    colnames(TempRowResNet152) <- c(
      "Precision",
      "Recall",
      "F1",
      "Validation loss",
      "Training Data",
      "N epochs",
      "CNN Architecture"
    )
    
    TempRowResNet152$Threshold <- as.character(threshold)
    
    CombinedTempRowThreshold <- rbind.data.frame(TempRowResNet18, TempRowResNet50, TempRowResNet152)
    CombinedTempRowThreshold$Threshold <- as.character(threshold)
    
    # Append to the overall result data frame
    CombinedTempRow <- rbind.data.frame(CombinedTempRow, CombinedTempRowThreshold)
  }
  
  # Append to the main data frame
  TransferLearningCNNDF <- rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)
  TransferLearningCNNDF$Frozen <- unfreeze.param
  # Write the result to a CSV file
  filename <- paste(output.data.path,'performance_tables/', trainingfolder, '_', n.epoch, '_', '_TransferLearningCNNDFResNet.csv', sep = '')
  write.csv(TransferLearningCNNDF, filename, row.names = FALSE)
  
  rm(modelResNet18Gibbon)
  rm(modelResNet50Gibbon)
  rm(modelResNet152Gibbon)
}




