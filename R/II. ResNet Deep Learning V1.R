# Load libraries ----------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(dplyr)
library(pins)
library(ggplot2)

library(tuneR)
library(seewave)
library(signal)
library(stringr)
library(ROCR)
library(caret)
library(MLmetrics)

# V2 Add modelResnet to script

# Train modelResnet -------------------------------------------------------------
device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
#device <- 'mps'

train_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    (function(x) x$to(device = device)) %>%
    #transform_random_resized_crop(size = c(224, 224)) %>%
    #transform_color_jitter() %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

valid_transforms <- train_transforms

test_transforms <- valid_transforms

# Labeled data from BLED detector
input.data <-  'data/images'

# Combined uses both
train_ds <- image_folder_dataset(
  file.path(input.data, "train"),
  transform = train_transforms)


valid_ds <- image_folder_dataset(
  file.path(input.data, "valid"),
  transform = valid_transforms)


test_ds <- image_folder_dataset(
   file.path(input.data, "test"),
   transform = test_transforms)


train_ds$.length()
valid_ds$.length()
test_ds$.length()

class_names <- train_ds$classes
length(class_names)
n.classes <- length(class_names)

batch_size <- 6

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = batch_size)
test_dl <- dataloader(test_ds, batch_size = batch_size)

train_dl$.length()
valid_dl$.length()
test_dl$.length()


# Plot images -------------------------------------------------------------
library(dplyr)
batch <- train_dl$.iter()$.next()
classes <- batch[[2]]
classes

images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)
images <- std * images + mean
images <- images * 255
images[images > 255] <- 255
images[images < 0] <- 0

par(mfcol = c(4,6), mar = rep(1, 4))

images %>%
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[as_array(classes)]) %>%
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})

# Prepare resnet modelResnet -----------------------------------------------------------
convnet <- nn_module(
  initialize = function() {
    self$model <- model_resnet50(pretrained = TRUE)
    for (par in self$parameters) {
      par$requires_grad_(FALSE)
    }
    self$model$fc <- nn_sequential(
      nn_linear(self$model$fc$in_features, 1024),
      nn_relu(),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, n.classes)
    )
  },
  forward = function(x) {
    self$model(x)
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  )

# rates_and_losses <- model %>% lr_finder(train_dl)
# rates_and_losses %>% plot()
n.epochs <- 20

fitted <- model %>%
  fit(train_dl, epochs = n.epochs, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 2),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = n.epochs,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"),
        luz_callback_model_checkpoint(path = "cpt_resnet/"),
        luz_callback_csv_logger("logs_resnet.csv")
      ),
      verbose = TRUE)

# Save model output
modelResnetGunshot <- fitted

luz_save(modelResnetGunshot, "modelResnetGunshot.pt")
modelResnetGunshot <- luz_load("modelResnetGunshot.pt")

# Test data metrics -------------------------------------------------------
test_ds <- image_folder_dataset(
  file.path(input.data, "test"),
  transform = test_transforms)

# Variable indicating the number of files
nfiles <- test_ds$.length()

# Load the test images
test_dl <- dataloader(test_ds, batch_size =nfiles)

# Predict the test files
output <- predict(modelResnetGunshot,test_dl)

# Return the index of the max values (i.e. which class)
PredMPS <- torch_argmax(output, dim = 2)

# Save to cpu
PredMPS <- as_array(torch_tensor(PredMPS,device = 'cpu'))

# Convert to a factor
predictedResnet <- as.factor(PredMPS)
print(predictedResnet)

# Calculate the probability associated with each class
Probability <- as_array(torch_tensor(nnf_softmax(output, dim = 2),device = 'cpu'))
#predictedResnet <- as.factor(ifelse(Probability[,1] > 0.9,1,2))

# Get the correct labels
correct <- as.factor(as.array(test_ds$samples[[2]]))

# Create a confusion matrix
caret::confusionMatrix(predictedResnet,correct, mode='everything')

# Calcuate the F1 value
f1_val <- MLmetrics::F1_Score(y_pred = predictedResnet,
                              y_true = correct)
f1_val #




