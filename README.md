Gunshot Detection with AlexNet and VGG16 using ‘torch for R’
================

<!-- README.md is generated from README.Rmd. Please edit that file -->

This repository contains code for training and evaluating a gunshot
detection model using the AlexNet and VGG16 convolutional neural network
architecture. The model is trained on the Imagenet dataset and
fine-tuned for the specific task of gunshot detection. The code utilizes
the ‘luz’ package in R for deep learning with torch.

## Dataset

The dataset used for training, validation, and testing is obtained from
sound clips of annotated PAM data, and includes labeled images of
gunshots and noise. The images are preprocessed using various
transformations, such as resizing, color jitter, and normalization, to
prepare them for input into the model.

## Training

The training process involves fine-tuning the pretrained AlexNet or
VGG16 models by training new classifier layers. The model is set up with
a binary cross-entropy loss function and the Adam optimizer. Training is
performed for 1-3 epochs or 20 epochs with early stopping enabled to
prevent overfitting.

The trained model is saved for future use.

## Evaluation

To evaluate the performance of the trained model, the test dataset is
used. The images in the test dataset are preprocessed in the same way as
during training. The model predicts the probability of an image
belonging to the positive class (gunshot) and calculates the F1 score as
the evaluation metric. Additionally, a confusion matrix is generated to
provide insights into the model’s performance.

## Usage

To use the trained model for predictions, load the saved model using the
‘luz_load’ function. Then, preprocess the input image(s) in the same way
as during training and pass them to the model for prediction. The output
will be the predicted class label or probability.

``` r
# Load libraries
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)
library(stringr)
library(caret)

# Load the saved model
modelAlexnetGunshot <- luz_load("_train_1_modelAlexnet.pt")

# Set the input directory for the test images
test.input <- 'data/images/test'

# Get the list of image files
imageFileShort <- list.files(test.input, recursive = TRUE, full.names = FALSE)

# Identify just the class from folder structure
ActualClass <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,1]

# Create a dataset from the image folder
test_ds <- image_folder_dataset(
  file.path(test.input),
  transform = . %>%
    torchvision::transform_to_tensor() %>%  # Convert images to tensors
    torchvision::transform_resize(size = c(224, 224)) %>%  # Resize images to 224x224 pixels
    torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)),  # Normalize the image tensor
  target_transform = function(x) as.double(x) - 1  # Transform the target variable to be zero-based
)

# Get the number of test files
nfiles <- test_ds$.length()

# Create a dataloader for the test dataset with a specified batch size
test_dl <- dataloader(test_ds, batch_size = nfiles)

# Use the Alexnet model to predict the test images
alexnetPred <- predict(modelAlexnetGunshot, test_dl)

# Apply the sigmoid function to the predicted values
alexnetProb <- torch_sigmoid(alexnetPred)

# Convert the predicted probabilities to an R array and subtract from 1
alexnetProb <- 1 - as_array(torch_tensor(alexnetProb, device = 'cpu'))

# Classify the images as "gunshot" or "noise" based on a probability threshold
alexnetPredictedClass <- ifelse((alexnetProb) > 0.85, "gunshot", "noise")

# Create a confusion matrix
AlexnetPerf <- caret::confusionMatrix(as.factor(alexnetPredictedClass),as.factor(ActualClass), mode='everything')

print(AlexnetPerf$table)
```

## Acknowledgments

This project was inspired by the work of Keydana (2023) on image
classification with transfer learning. Special thanks to the luz package
authors for providing the framework for deep learning in R with PyTorch.

## References

Keydana, Sigrid. Deep Learning and Scientific Computing with R torch.
CRC Press, 2023.
<https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/>
