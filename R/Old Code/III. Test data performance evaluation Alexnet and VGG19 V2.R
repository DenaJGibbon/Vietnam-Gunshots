
      # Load required libraries
      library(luz)
      library(torch)
      library(torchvision)
      library(stringr)
      library(ROCR)

      # Load pre-trained models
     #modelAlexnetGunshot <- luz_load("ModelsforExperiment/modelAlexnetGunshot1epoch.pt")
     #modelVGG19Gunshot <- luz_load("ModelsforExperiment/modelVGG19Gunshotearly1epoch.pt")

      # Set the path to image files
      imagePath <- "data/images/finaltest"

      # Get the list of image files
      imageFiles <- list.files(imagePath, recursive = TRUE, full.names = TRUE)

      # Get the list of image files
      imageFileShort <- list.files(imagePath, recursive = TRUE, full.names = FALSE)

      Folder <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,1]

      imageFileShort <- str_split_fixed( imageFileShort,pattern = '/',n=2)[,2]
      imageFileShort <- paste(Folder,imageFileShort,sep='')

      # Prepare output tables
      outputTableAlexnet <- data.frame()
      outputTableVGG19 <- data.frame()

      # Iterate over image files
      ImageFilesSeq <- seq(1,length(imageFiles),100)

      for (i in 1: (length(ImageFilesSeq)-1) ) {
        print(paste('processing', i, 'out of',length(ImageFilesSeq)))

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

        # Predict using VGG19
        VGG19Pred <- predict(modelVGG19Gunshot, test_dl)
        VGG19Prob <- torch_sigmoid(VGG19Pred)
        VGG19Prob <- 1-as_array(torch_tensor(VGG19Prob,device = 'cpu'))
        VGG19Class <- ifelse((VGG19Prob) > 0.85, "gunshot", "noise")

        # Add the results to output tables
        outputTableAlexnet <- rbind(outputTableAlexnet, data.frame(Label = TempLong, Probability = alexnetProb, PredictedClass = alexnetClass, ActualClass=Folder[ImageFilesSeq[i]: ImageFilesSeq[i+1]]))
        outputTableVGG19 <- rbind(outputTableVGG19, data.frame(Label = TempLong, Probability = VGG19Prob, PredictedClass = VGG19Class, ActualClass=Folder[ImageFilesSeq[i]: ImageFilesSeq[i+1]]))

        unlink('data/Temp/Images/Images', recursive = TRUE)
        dir.create('data/Temp/Images/Images')

        # Save the output tables as CSV files
        write.csv(outputTableAlexnet, "output_alexnet.csv", row.names = FALSE)
        write.csv(outputTableVGG19, "output_vgg19.csv", row.names = FALSE)

      }


      outputTableAlexnet$PredictedClass <-  ifelse(outputTableAlexnet$Probability > 0.85, "gunshot", "noise")
      outputTableVGG19$PredictedClass <-  ifelse(outputTableVGG19$Probability > 0.85, "gunshot", "noise")

      # Create a confusion matrix
      caret::confusionMatrix(as.factor(outputTableAlexnet$PredictedClass),as.factor(outputTableAlexnet$ActualClass), mode='everything')

      # Create a confusion matrix
      caret::confusionMatrix(as.factor(outputTableVGG19$PredictedClass),as.factor(outputTableVGG19$ActualClass), mode='everything')

      # Calcuate the F1 value
      f1_val <- MLmetrics::F1_Score(y_pred = outputTableAlexnet$PredictedClass,
                                    y_true = outputTableAlexnet$ActualClass)
      f1_val #

      # Calcuate the F1 value
      f1_val <- MLmetrics::F1_Score(y_pred = outputTableVGG19$PredictedClass,
                                    y_true = outputTableVGG19$ActualClass)
      f1_val #

