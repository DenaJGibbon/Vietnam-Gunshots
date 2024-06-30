# Combine training and testing into one script
# Packages ----------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)
library(stringr)
library(ROCR)

# Datasets ----------------------------------------------------------------

device <- if(cuda_is_available()) "cuda" else "cpu"

to_device <- function(x, device) {
  x$to(device = device)
}

# Spectrogram images for training
input.data <-  'data/images'

# Location of spectrogram images for testing
# Set the path to image files
imagePath <- "data/images/finaltest"

trainingfolders <- c("train", "trainaddedcleaned")
epoch.iterations <- c(5)

n.classes <- 2


for(a in 1:length(trainingfolders)){

  TransferLearningCNNDF <- data.frame()

  for(b in 1:length(epoch.iterations)){

    trainingfolder <- trainingfolders[a]
    n.epochs <- epoch.iterations [b]

# Combined uses both
train_ds <- image_folder_dataset(
  file.path(input.data,trainingfolder ),
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_color_jitter() %>%
    torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
  target_transform = function(x) as.double(x) - 1)

valid_ds <- image_folder_dataset(
  file.path(input.data, "valid"),
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
  target_transform = function(x) as.double(x) - 1)


train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)


# Train AlexNet -----------------------------------------------------------

net <- torch::nn_module(

  initialize = function() {
    self$model <- model_alexnet(pretrained = TRUE)

    for (par in self$parameters) {
      par$requires_grad_(TRUE)
    }

    self$model$classifier <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(9216, 512),
      nn_relu(),
      nn_linear(512, 256),
      nn_relu(),
      nn_linear(256, n.classes)
    )
  },
  forward = function(x) {
    self$model(x)[,1]
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


modelAlexnetGunshot <- fitted %>%
  fit(train_dl, epochs = n.epochs, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 2),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = n.epochs,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"),
        luz_callback_model_checkpoint(path = "cpt_Alexnet/"),
        luz_callback_csv_logger(paste( 'output_unfrozen/',trainingfolder,n.epochs, "logs_Alexnet.csv",sep='_'))
      ),
      verbose = TRUE)

# Save model output
luz_save(modelAlexnetGunshot, paste( 'output_unfrozen/',trainingfolder,n.epochs, "modelAlexnet.pt",sep='_'))
#modelAlexnetGunshot <- luz_load("modelAlexnetGunshot1epochs.pt")

TempCSV.Alexnet <-  read.csv(paste( 'output_unfrozen/',trainingfolder,n.epochs, "logs_Alexnet.csv",sep='_'))

Alexnet.loss <- TempCSV.Alexnet[nrow(TempCSV.Alexnet),]$loss


# Train VGG16 -------------------------------------------------------------

net <- torch::nn_module(
  initialize = function() {
    self$model <- model_vgg16 (pretrained = TRUE)

    for (par in self$parameters) {
      par$requires_grad_(TRUE)
    }

    self$model$classifier <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(25088, 4096),
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(4096, 4096),
      nn_relu(),
      nn_linear(4096, n.classes)
    )
  },
  forward = function(x) {
    self$model(x)[, 1]
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

modelVGG16Gunshot <- fitted %>%
  fit(train_dl, epochs = n.epochs, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 2),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = n.epochs,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"
        ),
        luz_callback_model_checkpoint(path = "cpt_VGG16/"),
        luz_callback_csv_logger(paste( 'output_unfrozen/',trainingfolder,n.epochs, "logs_VGG16.csv",sep='_'))
      ),
      verbose = TRUE)

# Save model output
luz_save(modelVGG16Gunshot, paste( 'output_unfrozen/',trainingfolder,n.epochs, "modelVGG16.pt",sep='_'))


TempCSV.VGG16 <- read.csv(paste( 'output_unfrozen/',trainingfolder,n.epochs, "logs_VGG16.csv",sep='_'))

VGG16.loss <- TempCSV.VGG16[nrow(TempCSV.VGG16),]$loss

# Calculate performance metrics -------------------------------------------

# Get the list of image files
imageFiles <- list.files(imagePath, recursive = TRUE, full.names = TRUE)

# Get the list of image files
imageFileShort <- list.files(imagePath, recursive = TRUE, full.names = FALSE)

Folder <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,1]

imageFileShort <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,2]
imageFileShort <- paste( Folder,imageFileShort,sep='')

# Prepare output tables
outputTableAlexnet <- data.frame()
outputTableVGG16 <- data.frame()

# Iterate over image files
ImageFilesSeq <- seq(1,length(imageFiles),100)

for (i in 1: (length(ImageFilesSeq)-1) ) {
  print(paste( 'output_unfrozen/','processing', i, 'out of',length(ImageFilesSeq)))

  batchSize <-  length(ImageFilesSeq)

  TempLong <- imageFiles[ ImageFilesSeq[i]: ImageFilesSeq[i+1]]
  TempShort <-  imageFileShort[  ImageFilesSeq[i]: ImageFilesSeq[i+1]]

  # Load and preprocess the image
  file.copy(TempLong,
            to= paste('data/Temp/Images/Images/',TempShort, sep=''))


  test.input <- 'data/Temp/Images/'

  test_ds <- image_folder_dataset(
    file.path(test.input),
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),
    target_transform = function(x) as.double(x) - 1)

  # Predict the test files
  # Variable indicating the number of files
  nfiles <- test_ds$.length()

  # Load the test images
  test_dl <- dataloader(test_ds, batch_size =batchSize)

  # Predict using Alexnet
  alexnetPred <- predict(modelAlexnetGunshot, test_dl)
  alexnetProb <- torch_sigmoid(alexnetPred)
  alexnetProb <- 1-as_array(torch_tensor(alexnetProb,device = 'cpu'))
  alexnetClass <- ifelse((alexnetProb) > 0.85, "gunshot", "noise")

  # # Predict using VGG16
  # VGG16Pred <- predict(modelVGG16Gunshot, test_dl)
  # VGG16Prob <- torch_sigmoid(VGG16Pred)
  # VGG16Prob <- 1-as_array(torch_tensor(VGG16Prob,device = 'cpu'))
  # VGG16Class <- ifelse((VGG16Prob) > 0.85, "gunshot", "noise")

  # Add the results to output tables
  outputTableAlexnet <- rbind(outputTableAlexnet, data.frame(Label = TempLong, Probability = alexnetProb, PredictedClass = alexnetClass, ActualClass=Folder[ImageFilesSeq[i]: ImageFilesSeq[i+1]]))
  # outputTableVGG16 <- rbind(outputTableVGG16, data.frame(Label = TempLong, Probability = VGG16Prob, PredictedClass = VGG16Class, ActualClass=Folder[ImageFilesSeq[i]: ImageFilesSeq[i+1]]))

  # Save false positives to train next iteration
  # file.copy(TempLong[which(alexnetProb > 0.9)],
  #           to= paste('data/Temp/Images/Images/',TempShort[which(alexnetProb > 0.9)], sep=''))

  unlink('data/Temp/Images/Images', recursive = TRUE)
  dir.create('data/Temp/Images/Images')

  # Save the output tables as CSV files

  write.csv(outputTableAlexnet, paste( 'output_unfrozen/',trainingfolder,n.epochs, "output_alexnet.csv",sep='_'), row.names = FALSE)
  #write.csv(outputTableVGG16, paste( 'output_unfrozen/',trainingfolder,n.epochs, "output_VGG16.csv",sep='_'), row.names = FALSE)

}


outputTableAlexnet$PredictedClass <-  ifelse(outputTableAlexnet$Probability > 0.95, "gunshot", "noise")
outputTableVGG16$PredictedClass <-  ifelse(outputTableVGG16$Probability > 0.95, "gunshot", "noise")

# Create a confusion matrix
AlexnetPerf <- caret::confusionMatrix(as.factor(outputTableAlexnet$PredictedClass),as.factor(outputTableAlexnet$ActualClass), mode='everything')$byClass

# Create a confusion matrix
VGG16Perf <- caret::confusionMatrix(as.factor(outputTableVGG16$PredictedClass),as.factor(outputTableVGG16$ActualClass), mode='everything')$byClass

TempRow95Alexnet <- cbind.data.frame( t(AlexnetPerf[c(5:7)]), Alexnet.loss,trainingfolder,n.epochs, 'Alexnet')

colnames(TempRow95Alexnet) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
                                "N epochs", "CNN Architecture")

TempRow95VGG16 <- cbind.data.frame( t(VGG16Perf[c(5:7)]), VGG16.loss,trainingfolder,n.epochs, 'VGG16')

colnames(TempRow95VGG16) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
                              "N epochs", "CNN Architecture")

CombinedTempRow95 <- rbind.data.frame(TempRow95Alexnet,TempRow95VGG16)
CombinedTempRow95$Threshold <- '95'

outputTableAlexnet$PredictedClass <-  ifelse(outputTableAlexnet$Probability > 0.85, "gunshot", "noise")
outputTableVGG16$PredictedClass <-  ifelse(outputTableVGG16$Probability > 0.85, "gunshot", "noise")

# Create a confusion matrix
AlexnetPerf <- caret::confusionMatrix(as.factor(outputTableAlexnet$PredictedClass),as.factor(outputTableAlexnet$ActualClass), mode='everything')$byClass

# Create a confusion matrix
VGG16Perf <- caret::confusionMatrix(as.factor(outputTableVGG16$PredictedClass),as.factor(outputTableVGG16$ActualClass), mode='everything')$byClass

TempRow85Alexnet <- cbind.data.frame( t(AlexnetPerf[c(5:7)]), Alexnet.loss,trainingfolder,n.epochs, 'Alexnet')

colnames(TempRow85Alexnet) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
  "N epochs", "CNN Architecture")

TempRow85VGG16 <- cbind.data.frame( t(VGG16Perf[c(5:7)]), VGG16.loss,trainingfolder,n.epochs, 'VGG16')

colnames(TempRow85VGG16) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
                            "N epochs", "CNN Architecture")

CombinedTempRow85 <- rbind.data.frame(TempRow85Alexnet,TempRow85VGG16)
CombinedTempRow85$Threshold <- '85'


outputTableAlexnet$PredictedClass <-  ifelse(outputTableAlexnet$Probability > 0.5, "gunshot", "noise")
outputTableVGG16$PredictedClass <-  ifelse(outputTableVGG16$Probability > 0.5, "gunshot", "noise")

# Create a confusion matrix
AlexnetPerf <- caret::confusionMatrix(as.factor(outputTableAlexnet$PredictedClass),as.factor(outputTableAlexnet$ActualClass), mode='everything')$byClass

# Create a confusion matrix
VGG16Perf <- caret::confusionMatrix(as.factor(outputTableVGG16$PredictedClass),as.factor(outputTableVGG16$ActualClass), mode='everything')$byClass

TempRow50Alexnet <- cbind.data.frame( t(AlexnetPerf[c(5:7)]), Alexnet.loss,trainingfolder,n.epochs, 'Alexnet')

colnames(TempRow50Alexnet) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
                                "N epochs", "CNN Architecture")

TempRow50VGG16 <- cbind.data.frame( t(VGG16Perf[c(5:7)]), VGG16.loss,trainingfolder,n.epochs, 'VGG16')

colnames(TempRow50VGG16) <- c("Precision", "Recall", "F1", "Validation loss", "Training Data",
                              "N epochs", "CNN Architecture")

CombinedTempRow50 <- rbind.data.frame(TempRow50Alexnet,TempRow50VGG16)
CombinedTempRow50$Threshold <- '50'

CombinedTempRow <- rbind.data.frame(CombinedTempRow95,CombinedTempRow85,CombinedTempRow50)

TransferLearningCNNDF <-rbind.data.frame(TransferLearningCNNDF, CombinedTempRow)

write.csv(TransferLearningCNNDF, paste('output_unfrozen/',trainingfolder, '_', n.epochs, '_TransferLearningCNNDF.csv',sep=''), row.names = F)

rm(modelAlexnetGunshot)
rm(modelVGG16Gunshot)
}
}
