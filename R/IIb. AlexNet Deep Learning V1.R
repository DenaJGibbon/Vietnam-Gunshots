# Packages ----------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

# Datasets ----------------------------------------------------------------

device <- if(cuda_is_available()) "cuda" else "cpu"

to_device <- function(x, device) {
  x$to(device = device)
}

# Labeled data from BLED detector
input.data <-  'data/images'

# Combined uses both
train_ds <- image_folder_dataset(
  file.path(input.data, "train"),
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
    target_transform = function(x) as.double(x) - 1)

valid_ds <- image_folder_dataset(
  file.path(input.data, "valid"),
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
  target_transform = function(x) as.double(x) - 1)


train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)

class_names <- train_ds$classes
length(class_names)
n.classes <- length(class_names)

net <- torch::nn_module(

  initialize = function() {
    self$model <- model_alexnet(pretrained = TRUE)

    for (par in self$parameters) {
      par$requires_grad_(FALSE)
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

n.epochs <- 20

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
        luz_callback_csv_logger("logs_Alexnet.csv")
      ),
      verbose = TRUE)

# Save model output
luz_save(modelAlexnetGunshot, "modelAlexnetGunshot.pt")
modelAlexnetGunshot <- luz_load("modelAlexnetGunshot.pt")

# Test data metrics -------------------------------------------------------
test_ds <- image_folder_dataset(
  file.path(input.data, "test"),
  transform = . %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_resize(size = c(224, 224)) %>%
    torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
    target_transform = function(x) as.double(x) - 1)

# Variable indicating the number of files
nfiles <- test_ds$.length()

# Load the test images
test_dl <- dataloader(test_ds, batch_size =nfiles)

# Predict the test files

preds <- predict(modelAlexnetGunshot, test_dl)

# Probability of being in class 1
probs <- torch_sigmoid(preds)

PredMPS <- 1- as_array(torch_tensor(probs,device = 'cpu'))

predictedAlexnet <- as.factor(ifelse(PredMPS > 0.5,1,2))

# Get the correct labels
correct <- as.factor(as.array(test_ds$samples[[2]]))

# Create a confusion matrix
caret::confusionMatrix(predictedAlexnet,correct, mode='everything')

# Calcuate the F1 value
f1_val <- MLmetrics::F1_Score(y_pred = predictedAlexnet,
                              y_true = correct)
f1_val #


# ROCR curves -------------------------------------------------------------
pred <- prediction( PredMPS, correct)
pred
perf <- performance(pred,"tpr","fpr")
perf
plot(perf)

