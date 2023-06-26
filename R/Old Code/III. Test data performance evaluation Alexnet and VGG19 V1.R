# Prepare data ------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(stringr)
library(tuneR)
library(seewave)
library(gibbonR)

# Load pre-trained models
modelAlexnetGunshot <- luz_load("modelAlexnetGunshot.pt")
modelVGG19Gunshot <- luz_load("modelVGG19Gunshot.pt")

Fullpath.images <- list.files('data/images/finaltest',recursive = T,full.names = T)
Shortpath.images <- list.files('data/images/finaltest',recursive = T,full.names = T)

Folder <- str_split_fixed(Shortpath.images, pattern = '/',n=5)[,4]
Shortpath.images.orig <- str_split_fixed(Shortpath.images, pattern = '/',n=5)[,5]
Shortpath.images <- paste(Folder,Shortpath.images.orig, sep='_')

clip.duration <- 6
hop.size <- 3

# Subset sound clips for classification -----------------------------------
      print('saving sound clips')
      set.seed(13)
      length <- length(Shortpath.images)
      length.files <- seq(1,length,100)

      RavenSelectionTableDFAlexnet <- data.frame()
      RavenSelectionTableDFVGG19 <- data.frame()

      for(q in 1: (length(length.files)-1) ){

        RandomSub <-  seq(length.files[q],length.files[q+1],1)

        # Save images to a temp folder
        print('Copying images')
        file.copy(Fullpath.images[RandomSub],
                  to= paste('data/Temp/Images/Images/',
                            Shortpath.images[RandomSub],
                            '_',
                            '.jpg', sep=''))


    # Predict using Alexnet ----------------------------------------------------
        print('Classifying images using Alexnet')

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
        test_dl <- dataloader(test_ds, batch_size =nfiles)

        preds <- predict(modelAlexnetGunshot, test_dl)

        # Probability of being in class 1
        probs <- torch_sigmoid(preds)

        PredMPS <- as_array(torch_tensor(probs,device = 'cpu'))

        PredMPS <- 1-PredMPS

        predictedAlexnet <- as.factor(ifelse(PredMPS > 0.5,1,2))

        # Calculate the probability associated with each class
        Probability <- PredMPS

        OutputFolder <- 'data/Detections/Alexnet/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]


        print('Saving output')
        DetectionsIndex <-  RandomSub


        # file.copy(image.files[Detections],
        #           to= paste(OutputFolder,
        #                     image.files.short[Detections],
        #                     '_',
        #                     round(Probability,2),
        #                     '_Alexnet_.jpg', sep=''))

        Detections <- image.files.short[DetectionsIndex]


        if (length(Detections) > 0) {
          Selection <- seq(1, length(Detections))
          View <- rep('Spectrogram 1', length(Detections))
          Channel <- rep(1, length(Detections))
          MinFreq <- rep(100, length(Detections))
          MaxFreq <- rep(2000, length(Detections))
          Probability <- round(Probability,2)
          Class <- Folder[DetectionsIndex]

          RavenSelectionTableDFAlexnetTemp <-
            cbind.data.frame(Selection,
                             View,
                             Channel,
                             MinFreq,
                             MaxFreq,Probability,Class,
                             Detections)

          RavenSelectionTableDFAlexnetTemp <-
            RavenSelectionTableDFAlexnetTemp[, c(
              "Selection",
              "View",
              "Channel",
              "MinFreq",
              "MaxFreq",
              'Probability',"Class","Detections"
            )]


          RavenSelectionTableDFAlexnet <- rbind.data.frame(RavenSelectionTableDFAlexnet,
                                                    RavenSelectionTableDFAlexnetTemp)

          if(nrow(RavenSelectionTableDFAlexnet) > 0){
            csv.file.name <-
              paste('data/',
                    'TestDataSelection',
                    'GunshotAlexNET.txt',
                    sep = '')

            write.table(
              x = RavenSelectionTableDFAlexnet,
              sep = "\t",
              file = csv.file.name,
              row.names = FALSE,
              quote = FALSE
            )
            print(paste(
              "Saving Selection Table"
            ))
          }

          # Predict using VGG19 ----------------------------------------------------
          print('Classifying images using VGG19')

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
          test_dl <- dataloader(test_ds, batch_size =nfiles)

          preds <- predict(modelVGG19Gunshot, test_dl)

          # Probability of being in class 1
          probs <- torch_sigmoid(preds)

          PredMPS <- as_array(torch_tensor(probs,device = 'cpu'))

          PredMPS <- 1-PredMPS

          predictedVGG19 <- as.factor(ifelse(PredMPS > 0.5,1,2))

          # Calculate the probability associated with each class
          Probability <- PredMPS

          OutputFolder <- 'data/Detections/VGG19/'

          image.files <- list.files(file.path(test.input),recursive = T,
                                    full.names = T)
          nslash <- str_count(image.files,'/')+1
          nslash <- nslash[1]
          image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
          image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

          print('Saving output')
          Detections <-  which(Probability > 0 )

          print('Saving output')
          # file.copy(image.files[Detections],
          #           to= paste(OutputFolder,
          #                     image.files.short[Detections],
          #                     '_',
          #                     round(Probability,2),
          #                     '_VGG19_.jpg', sep=''))

          #Detections <- image.files.short[Detections]


          if (length(Detections) > 0) {
            Selection <- seq(1, length(Detections))
            View <- rep('Spectrogram 1', length(Detections))
            Channel <- rep(1, length(Detections))
            MinFreq <- rep(100, length(Detections))
            MaxFreq <- rep(2000, length(Detections))
            start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=3)[,3])
            end.time.new <- start.time.new + clip.duration
            Probability <- round(Probability,2)
            Class <-

            RavenSelectionTableDFVGG19Temp <-
              cbind.data.frame(Selection,
                               View,
                               Channel,
                               MinFreq,
                               MaxFreq,start.time.new,end.time.new,Probability,
                               Detections)

            RavenSelectionTableDFVGG19Temp <-
              RavenSelectionTableDFVGG19Temp[, c(
                "Selection",
                "View",
                "Channel",
                "start.time.new",
                "end.time.new",
                "MinFreq",
                "MaxFreq",
                'Probability',"Detections"
              )]

            colnames(RavenSelectionTableDFVGG19Temp) <-
              c(
                "Selection",
                "View",
                "Channel",
                "Begin Time (s)",
                "End Time (s)",
                "Low Freq (Hz)",
                "High Freq (Hz)",
                'Probability',
                "Detections"
              )

            RavenSelectionTableDFVGG19 <- rbind.data.frame(RavenSelectionTableDFVGG19,
                                                      RavenSelectionTableDFVGG19Temp)

            if(nrow(RavenSelectionTableDFVGG19) > 0){
              csv.file.name <-
                paste('data/',
                      'TestDataSelection',
                      'GunshotVGG19.txt',
                      sep = '')

              write.table(
                x = RavenSelectionTableDFVGG19,
                sep = "\t",
                file = csv.file.name,
                row.names = FALSE,
                quote = FALSE
              )
              print(paste(
                "Saving Selection Table"
              ))

          }
        }
      }

        unlink('data/Temp/WavFiles', recursive = TRUE)
        unlink('data/Temp/Images/Images', recursive = TRUE)

        dir.create('data/Temp/WavFiles')
        dir.create('data/Temp/Images/Images')
      }


      # Load required libraries
      library(luz)
      library(torch)
      library(torchvision)
      library(stringr)

      # Load pre-trained models
      modelAlexnetGunshot <- luz_load("modelAlexnetGunshot.pt")
      modelVGG19Gunshot <- luz_load("modelVGG19Gunshot.pt")

      # Set the path to image files
      imagePath <- "data/images/finaltest"

      # Get the list of image files
      imageFiles <- list.files(imagePath, recursive = TRUE, full.names = TRUE)

      # Prepare output tables
      outputTableAlexnet <- data.frame()
      outputTableVGG19 <- data.frame()

      # Iterate over image files
      for (file in imageFiles) {
        # Load and preprocess the image
        img <- jpeg::readJPEG(file)
        imgTensor <- torchvision::transform_to_tensor(img)
        imgTensor <- torchvision::transform_normalize(imgTensor,mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
        imgTensor <-  torchvision::transform_resize(imgTensor,size = c(224, 224))

        # Predict using Alexnet
        alexnetPred <- predict(modelAlexnetGunshot, imgTensor)
        alexnetProb <- torch_sigmoid(alexnetPred)
        alexnetClass <- ifelse(as_array(alexnetProb) > 0.5, "Class 1", "Class 2")

        # Predict using VGG19
        vgg19Pred <- predict(modelVGG19Gunshot, img)
        vgg19Prob <- torch_sigmoid(vgg19Pred)
        vgg19Class <- ifelse(as_array(vgg19Prob) > 0.5, "Class 1", "Class 2")

        # Add the results to output tables
        outputTableAlexnet <- rbind(outputTableAlexnet, data.frame(Label = file, Probability = as_array(alexnetProb), Class = alexnetClass))
        outputTableVGG19 <- rbind(outputTableVGG19, data.frame(Label = file, Probability = as_array(vgg19Prob), Class = vgg19Class))
      }

      # Save the output tables as CSV files
      write.csv(outputTableAlexnet, "output_alexnet.csv", row.names = FALSE)
      write.csv(outputTableVGG19, "output_vgg19.csv", row.names = FALSE)
