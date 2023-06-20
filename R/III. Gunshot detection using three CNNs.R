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
modelResnetGunshot <- luz_load("modelResnetGunshot.pt")
modelAlexnetGunshot <- luz_load("modelAlexnetGunshot.pt")
modelVGG19 <- luz_load("modelVGG19.pt")

# Set path to BoxDrive
BoxDrivePath <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                           full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

SoundFileDF <- data.frame()

clip.duration <- 6
hop.size <- 3

RavenSelectionTableDF <- data.frame()


    for(x in 1:length(BoxDrivePath)){ tryCatch({
      start.time <- Sys.time()
      print(paste(x, 'out of', length(BoxDrivePath)))
      TempWav <- readWave(BoxDrivePath[x])
      WavDur <- duration(TempWav)

      Seq.start <- list()
      Seq.end <- list()

      i <- 1
      while (i + clip.duration < WavDur) {
        # print(i)
        Seq.start[[i]] = i
        Seq.end[[i]] = i+clip.duration
        i= i+hop.size
      }

      ClipStart <- unlist(Seq.start)
      ClipEnd <- unlist(Seq.end)

      TempClips <- cbind.data.frame(ClipStart,ClipEnd)

      short.sound.files <- lapply(1:nrow(TempClips),
                                  function(i)
                                    extractWave(
                                      TempWav,
                                      from = TempClips$ClipStart[i],
                                      to = TempClips$ClipEnd[i],
                                      xunit = c("time"),
                                      plot = F,
                                      output = "Wave"
                                    ))

      # Save .wav to a temp folder
      print('saving sound clips')
      set.seed(13)
      length <- length(short.sound.files)
      length.files <- seq(1,length,100)

      for(q in 1: (length(length.files)-1) ){

        RandomSub <-  seq(length.files[q],length.files[q+1],1)
        start.time <- TempClips$ClipStart[RandomSub]
        end.time <- TempClips$ClipEnd[RandomSub]

        for(d in RandomSub){
          #print(d)
          writeWave(short.sound.files[[d]],paste('data/Temp/WavFiles','/',
                                                 BoxDrivePathShort[x],'_',start.time[d], '.wav', sep=''),
                    extensible = F)
        }

        # Save images to a temp folder
        print('Creating images')
        for(e in RandomSub){
          jpeg(paste('data/Temp/Images/Images','/', BoxDrivePathShort[x],'_',TempClips$ClipStart[e],'.jpg',sep=''),res = 50)
          short.wav <- short.sound.files[[e]]

          seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0,2),fastdisp=TRUE,noisereduction=1)

          graphics.off()
        }




        # Predict using ResNET ----------------------------------------------------
        print('Classifying images using ResNET')
        test.input <- 'data/Temp/Images/'
        test_ds <- image_folder_dataset(
          file.path(test.input),
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
        #predictedResnet <- as.factor(ifelse(Probability[,1] > 0.5,1,2))

        OutputFolder <- 'data/Detections/Resnet/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

        print('Saving output')
        file.copy(image.files[which(predictedResnet==1)],
                  to= paste(OutputFolder,
                            image.files.short[which(predictedResnet==1 )],
                            '_',
                            round(Probability[which(predictedResnet==1 )],2),
                            '_resnet_.jpg', sep=''))

        #Start audio classification
        DetectionsResnet <-  image.files.short[which(predictedResnet==1 )]

        if(length(DetectionsResnet) > 0){
          TempRowsResnet <- cbind.data.frame(DetectionsResnet,'Resnet',round(Probability[which(predictedResnet==1 )],2))
          colnames(TempRowsResnet) <- c('Detection','Model','Probability')
        } else {
          TempRowsResnet <- cbind.data.frame('NA','Resnet','NA')
          colnames(TempRowsResnet) <- c('Detection','Model','Probability')

        }
        # Predict using Alexnet ----------------------------------------------------
        print('Classifying images using Alexnet')

        test.input <- 'data/Temp/Images/'

        test_ds <- image_folder_dataset(
          file.path(test.input),
          transform = . %>%
            torchvision::transform_to_tensor() %>%
            torchvision::transform_resize(size = c(224, 224)) %>%
            torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
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

        predictedAlexnet <- as.factor(ifelse(PredMPS < 0.5,1,2))

        # Calculate the probability associated with each class
        Probability <- 1-PredMPS

        OutputFolder <- 'data/Detections/Alexnet/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

        print('Saving output')
        file.copy(image.files[which(predictedAlexnet==1)],
                  to= paste(OutputFolder,
                            image.files.short[which(predictedAlexnet==1 )],
                            '_',
                            round(Probability[which(predictedAlexnet==1 )],2),
                            '_Alexnet_.jpg', sep=''))

        #Start audio classification
        DetectionsAlexnet <-  image.files.short[which(predictedAlexnet==1 )]

        if(length(DetectionsAlexnet) > 0){
          TempRowsAlexnet <- cbind.data.frame(DetectionsAlexnet,'Alexnet',round(Probability[which(predictedAlexnet==1 )],2))
          colnames(TempRowsAlexnet) <- c('Detection','Model','Probability')
        } else {
          TempRowsAlexnet <- cbind.data.frame('NA','Alexnet','NA')
          colnames(TempRowsAlexnet) <- c('Detection','Model','Probability')

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
          target_transform = function(x) as.double(x) - 1
        )

        # Variable indicating the number of files
        nfiles <- test_ds$.length()

        # Load the test images
        test_dl <- dataloader(test_ds, batch_size =nfiles)

        preds <- predict(modelVGG19, test_dl)

        # Probability of being in class 1
        probs <- torch_sigmoid(preds)

        PredMPS <- as_array(torch_tensor(probs,device = 'cpu'))

        predictedVGG19 <- as.factor(ifelse(PredMPS < 0.5,1,2))

        # Calculate the probability associated with each class
        Probability <- 1-PredMPS

        OutputFolder <- 'data/Detections/VGG19/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

        print('Saving output')
        file.copy(image.files[which(predictedVGG19==1)],
                  to= paste(OutputFolder,
                            image.files.short[which(predictedVGG19==1 )],
                            '_',
                            round(Probability[which(predictedVGG19==1 )],2),
                            '_vgg19_.jpg', sep=''))

        #Start audio classification
        DetectionsVGG19 <-  image.files.short[which(predictedVGG19==1 )]

        if(length(DetectionsVGG19) > 0){
          TempRowsVGG19 <- cbind.data.frame(DetectionsVGG19,'VGG19',round(Probability[which(predictedVGG19==1 )],2))
          colnames(TempRowsVGG19) <- c('Detection','Model','Probability')
        } else {
          TempRowsVGG19 <- cbind.data.frame('NA','VGG19','NA')
          colnames(TempRowsVGG19) <- c('Detection','Model','Probability')

        }

        Detections <- rbind.data.frame(TempRowsResnet,TempRowsAlexnet,TempRowsVGG19)

        if (nrow(Detections) > 0) {
          Detections <- na.omit(Detections)
          Selection <- seq(1, nrow(Detections))
          View <- rep('Spectrogram 1', nrow(Detections))
          Channel <- rep(1, nrow(Detections))
          MinFreq <- rep(100, nrow(Detections))
          MaxFreq <- rep(2000, nrow(Detections))
          start.time.new <- as.numeric(str_split_fixed(Detections$Detection,pattern = '_',n=3)[,3])
          end.time.new <- start.time.new + clip.duration
          Probability <- Detections$Probability

          RavenSelectionTableDFTemp <-
            cbind.data.frame(Selection,
                             View,
                             Channel,
                             MinFreq,
                             MaxFreq,start.time.new,end.time.new,
                             Detections)

          RavenSelectionTableDFTemp <-
            RavenSelectionTableDFTemp[, c(
              "Selection",
              "View",
              "Channel",
              "start.time.new",
              "end.time.new",
              "MinFreq",
              "MaxFreq",
              'Probability',"Detection", "Model"
            )]

          colnames(RavenSelectionTableDFTemp) <-
            c(
              "Selection",
              "View",
              "Channel",
              "Begin Time (s)",
              "End Time (s)",
              "Low Freq (Hz)",
              "High Freq (Hz)",
              'Probability',
              "Detection", "Model"
            )

          RavenSelectionTableDF <- rbind.data.frame(RavenSelectionTableDF,
                                                    RavenSelectionTableDFTemp)




        }

        unlink('data/Temp/WavFiles', recursive = TRUE)
        unlink('data/Temp/Images/Images', recursive = TRUE)

        dir.create('data/Temp/WavFiles')
        dir.create('data/Temp/Images/Images')
      }

      if(nrow(RavenSelectionTableDF) > 0){
        RavenSelectionTableDF <- na.omit(RavenSelectionTableDF)
        csv.file.name <-
          paste('data/',
                BoxDrivePathShort[x],
                'GibbonResNET.txt',
                sep = '')

        write.table(
          x = RavenSelectionTableDFTemp,
          sep = "\t",
          file = csv.file.name,
          row.names = FALSE,
          quote = FALSE
        )
        print(paste(
          "Saving Selection Table"
        ))
      }

      rm(TempWav)
      rm(short.sound.files)
      rm( test_ds )
      rm(short.wav)
      end.time <- Sys.time()
      print(end.time-start.time)
    }, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
    }



