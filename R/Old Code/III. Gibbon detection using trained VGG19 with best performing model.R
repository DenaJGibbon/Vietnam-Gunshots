# Prepare data ------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(stringr)
library(tuneR)
library(seewave)
library(gibbonR)

# Note: This can be used to run over entire sound files

# Load pre-trained models

modelVGG19Gibbon <- luz_load("output_frozen/_train_4_modelVGG19.pt")
modelAlexNetGibbon <- luz_load("output_frozen/_train_2_modelAlexnet.pt")

clip.duration <- 8
hop.size <- 4

# Set path to BoxDrive
BoxDrivePath <- '/Users/denaclink/Library/CloudStorage/Box-Box/Cambodia 2022/'

Directories <- list.files(BoxDrivePath,full.names = T)
Directories <- Directories[-c(1,6,7)]

WavFilesDF <- data.frame()

for(a in 1:length(Directories)){
  WavList <- list.files(Directories[a],full.names = T,recursive = T, pattern = '.wav')
  nslash <- str_count(WavList[1],pattern = '/')+1
  ShortNames <- str_split_fixed(WavList,pattern = '/',n=nslash)[,nslash]
  Times <- str_split_fixed(ShortNames, pattern = '_',n=3)[,3]
  Times <- as.numeric(str_split_fixed(Times, pattern = '.wav',n=2)[,1])
  WavListAM <- WavList[which(Times >=50000 & Times <=70005)]
  WavListAMShort <- ShortNames[which(Times >=50000 & Times <=70005)]
  WavListAMShortNoWav <- str_split_fixed(WavListAMShort,pattern = '.wav',n=2)[,1]


    for(x in rev(1:length(WavListAM)) ){ tryCatch({
      RavenSelectionTableDFVGG19 <- data.frame()
      RavenSelectionTableDFAlexNet <- data.frame()
      start.time.detection <- Sys.time()
      print(paste('processing',x, 'out of', length(WavListAM), ' soundfiles for', a, 'out of', length(Directories), 'directories' ) )

      TempWav <- readWave(WavListAM[x])
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



# Subset sound clips for classification -----------------------------------
      print('saving sound clips')
      set.seed(13)
      length <- nrow(TempClips)
      length.files <- seq(1,length,100)

      for(q in 1: (length(length.files)-1) ){

        # Remove previous wav and image files for detection
        unlink('data/Temp/WavFiles', recursive = TRUE)
        unlink('data/Temp/Images/Images', recursive = TRUE)

        # Add new directory
        dir.create('data/Temp/WavFiles')
        dir.create('data/Temp/Images/Images')

        RandomSub <-  seq(length.files[q],length.files[q+1],1)

        if(q== (length(length.files)-1) ){
          RandomSub <-  seq(length.files[q],length,1)
        }

        start.time <- TempClips$ClipStart[RandomSub]
        end.time <- TempClips$ClipEnd[RandomSub]

        short.sound.files <- lapply(1:length(start.time),
                                    function(i)
                                      extractWave(
                                        TempWav,
                                        from = start.time[i],
                                        to = end.time[i],
                                        xunit = c("time"),
                                        plot = F,
                                        output = "Wave"
                                      ))

        for(d in 1:length(short.sound.files)){
          #print(d)
          writeWave(short.sound.files[[d]],paste('data/Temp/WavFiles','/',
                                                 WavListAMShortNoWav[x],'_',start.time[d], '.wav', sep=''),
                    extensible = F)
        }

        # Save images to a temp folder
        print(paste('Creating images',start.time[1],'start time clips'))

        for(e in 1:length(short.sound.files)){
          jpeg(paste('data/Temp/Images/Images','/', WavListAMShortNoWav[x],'_',start.time[e],'.jpg',sep=''),res = 50)
          short.wav <- short.sound.files[[e]]

          seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0,3),fastdisp=TRUE,noisereduction=1)

          graphics.off()
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

        # Predict using VGG19
        VGG19Pred <- predict(modelVGG19Gibbon, test_dl)
        VGG19Prob <- torch_sigmoid(VGG19Pred)
        VGG19Prob <- 1-as_array(torch_tensor(VGG19Prob,device = 'cpu'))
        VGG19Class <- ifelse((VGG19Prob) > 0.5, "Gibbons", "Noise")


        # Calculate the probability associated with each class
        Probability <- VGG19Prob

        OutputFolder <- 'data/Detections/VGG19/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

        print('Saving output')
        Detections <-  which(Probability > 0.5 )

        if(length(Detections) > 1){
        Detections <-  split(Detections, cumsum(c(
          1, diff(Detections)) != 1))

        for(i in 1:length(Detections)){
          TempList <- Detections[[i]]
          if(length(TempList)==1){
            Detections[[i]] <- TempList[1]
          }
          if(length(TempList)==2){
            Detections[[i]] <- TempList[2]
          }
          if(length(TempList)> 2){
            Detections[[i]] <- median(TempList)
          }

        }

        DetectionIndices <- unname(unlist(Detections))

        } else{
          DetectionIndices <- Detections
        }


        print('Saving output')
        file.copy(image.files[DetectionIndices],
                  to= paste(OutputFolder,
                            image.files.short[DetectionIndices],
                            '_',
                            round(Probability[DetectionIndices],2),
                            '_VGG19_.jpg', sep=''))

        Detections <- image.files.short[DetectionIndices]


        if (length(Detections) > 0) {
          Selection <- seq(1, length(Detections))
          View <- rep('Spectrogram 1', length(Detections))
          Channel <- rep(1, length(Detections))
          MinFreq <- rep(100, length(Detections))
          MaxFreq <- rep(3000, length(Detections))
          start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=4)[,4])
          end.time.new <- start.time.new + clip.duration
          Probability <- round(Probability[DetectionIndices],2)

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
              paste('data/selectiontables/',
                    WavListAMShortNoWav[x],
                    'GibbonsVGG19FinalPerformance.txt',
                    sep = '')

            write.table(
              x = RavenSelectionTableDFVGG19,
              sep = "\t",
              file = csv.file.name,
              row.names = FALSE,
              quote = FALSE
            )
            print(paste(
              "Saving Selection Table with Detections"
            ))
          }

        }
          # Predict using AlexNet ----------------------------------------------------
          print('Classifying images using AlexNet')

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

          # Predict using AlexNet
          AlexNetPred <- predict(modelAlexNetGibbon, test_dl)
          AlexNetProb <- torch_sigmoid(AlexNetPred)
          AlexNetProb <- 1-as_array(torch_tensor(AlexNetProb,device = 'cpu'))
          AlexNetClass <- ifelse((AlexNetProb) > 0.5, "Gibbons", "Noise")


          # Calculate the probability associated with each class
          Probability <- AlexNetProb

          OutputFolder <- 'data/Detections/AlexNet/'

          image.files <- list.files(file.path(test.input),recursive = T,
                                    full.names = T)
          nslash <- str_count(image.files,'/')+1
          nslash <- nslash[1]
          image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
          image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

          print('Saving output')
          Detections <-  which(Probability > 0.85 )

          if(length(Detections) > 1){

            Detections <-  split(Detections, cumsum(c(
              1, diff(Detections)) != 1))

            for(i in 1:length(Detections)){
              TempList <- Detections[[i]]
              if(length(TempList)==1){
                Detections[[i]] <- TempList[1]
              }
              if(length(TempList)==2){
                Detections[[i]] <- TempList[2]
              }
              if(length(TempList)> 2){
                Detections[[i]] <- median(TempList)
              }

            }

            DetectionIndices <- unname(unlist(Detections))

          } else{
            DetectionIndices <- Detections
          }



          print('Saving output')
          file.copy(image.files[DetectionIndices],
                    to= paste(OutputFolder,
                              image.files.short[DetectionIndices],
                              '_',
                              round(Probability[DetectionIndices],2),
                              '_AlexNet_.jpg', sep=''))

          Detections <- image.files.short[DetectionIndices]


          if (length(Detections) > 0) {
            Selection <- seq(1, length(Detections))
            View <- rep('Spectrogram 1', length(Detections))
            Channel <- rep(1, length(Detections))
            MinFreq <- rep(100, length(Detections))
            MaxFreq <- rep(3000, length(Detections))
            start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=4)[,4])
            end.time.new <- start.time.new + clip.duration
            Probability <- round(Probability[DetectionIndices],2)

            RavenSelectionTableDFAlexNetTemp <-
              cbind.data.frame(Selection,
                               View,
                               Channel,
                               MinFreq,
                               MaxFreq,start.time.new,end.time.new,Probability,
                               Detections)

            RavenSelectionTableDFAlexNetTemp <-
              RavenSelectionTableDFAlexNetTemp[, c(
                "Selection",
                "View",
                "Channel",
                "start.time.new",
                "end.time.new",
                "MinFreq",
                "MaxFreq",
                'Probability',"Detections"
              )]

            colnames(RavenSelectionTableDFAlexNetTemp) <-
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

            RavenSelectionTableDFAlexNet <- rbind.data.frame(RavenSelectionTableDFAlexNet,
                                                           RavenSelectionTableDFAlexNetTemp)

            if(nrow(RavenSelectionTableDFAlexNet) > 0){
              csv.file.name <-
                paste('data/selectiontables/',
                      WavListAMShortNoWav[x],
                      'GibbonsAlexNetFinalPerformance.txt',
                      sep = '')

              write.table(
                x = RavenSelectionTableDFAlexNet,
                sep = "\t",
                file = csv.file.name,
                row.names = FALSE,
                quote = FALSE
              )
              print(paste(
                "Saving Selection Table with Detections"
              ))
            }
          }
        }

      WavName <- WavListAMShortNoWav[x]
      end.time.detection <- Sys.time()
      Process.Time <- (end.time.detection-start.time.detection)
      TempRowWavDF <- cbind.data.frame(WavName,Process.Time)
      WavFilesDF <- rbind.data.frame(WavFilesDF,TempRowWavDF)
      write.csv(WavFilesDF,'data/WavFilesDFProcessed.csv', row.names = F)
      rm(TempWav)
      rm(short.sound.files)
      rm( test_ds )
      rm(short.wav)
      end.time.detection <- Sys.time()
      print(end.time.detection-start.time.detection)
    }, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
    }

}

