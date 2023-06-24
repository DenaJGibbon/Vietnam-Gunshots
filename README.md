Gunshot Detection with AlexNet and VGG19 using ‘torch for R’
================

<!-- README.md is generated from README.Rmd. Please edit that file -->

This repository contains code for training and evaluating a gunshot
detection model using the AlexNet and VGG19 convolutional neural network
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
VGG19 model by freezing the feature extraction layers and training new
classifier layers. The model is set up with a binary cross-entropy loss
function and the Adam optimizer. Training is performed for 20 epochs
with early stopping enabled to prevent overfitting. Additionally, a
learning rate scheduler is used to adjust the learning rate during
training.

The trained model is saved for future use.

## Evaluation

To evaluate the performance of the trained model, the test dataset is
used. The images in the test dataset are preprocessed in the same way as
during training. The model predicts the probability of an image
belonging to the positive class (gunshot) and calculates the F1 score as
the evaluation metric. Additionally, a confusion matrix is generated to
provide insights into the model’s performance.

## Usage

To use the trained model for inference, load the saved model using the
‘luz_load’ function. Then, preprocess the input image(s) in the same way
as during training and pass them to the model for prediction. The output
will be the predicted class label or probability.

``` r
# Load the saved model
model <- luz_load("modelAlexnetGunshot.pt")

# Preprocess the input image(s)
# ...

# Pass the preprocessed image(s) to the model for prediction
predictions <- predict(model, input_images)

# Process the predictions
# ...
```

## Acknowledgments

This project was inspired by the work of Keydana (2023) on image
classification with transfer learning. Special thanks to the luz package
authors for providing the framework for deep learning in R with PyTorch.

## References

Keydana, Sigrid. Deep Learning and Scientific Computing with R torch.
CRC Press, 2023.
<https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/>
