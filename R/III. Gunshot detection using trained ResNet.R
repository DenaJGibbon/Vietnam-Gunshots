# Prepare data ------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(stringr)
library(tuneR)
library(seewave)
library(gibbonR)

# Load pre-trained model
modelResnetGunshot <- luz_load("modelResnetGunshot.pt")

# Set path to BoxDrive
BoxDrivePath <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                           full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

SoundFileDF <- data.frame()

clip.duration <- 12
hop.size <- 6

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

        OutputFolder <- 'data/Detections/'

        image.files <- list.files(file.path(test.input),recursive = T,
                                  full.names = T)
        nslash <- str_count(image.files,'/')+1
        nslash <- nslash[1]
        image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
        image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]

        print('Saving output')
        #ProbabilityVal <- 0.6
        file.copy(image.files[which(predictedResnet==1)],
                  to= paste(OutputFolder,
                            image.files.short[which(predictedResnet==1 )],
                            '_',
                            round(Probability[which(predictedResnet==1 )],2),
                            '.jpg', sep=''))

        #Start audio classification
        Detections <-  image.files.short[which(predictedResnet==1 )]

        if (length(Detections) > 0) {

          Selection <- seq(1, length(Detections))
          View <- rep('Spectrogram 1', length(Detections))
          Channel <- rep(1, length(Detections))
          MinFreq <- rep(400, length(Detections))
          MaxFreq <- rep(3000, length(Detections))
          start.time <- start.time[which(predictedResnet==1 )]
          end.time <- end.time[which(predictedResnet==1 )]
          Probability <- round(Probability[which(predictedResnet==1 )],2)

          RavenSelectionTableDFTemp <-
            cbind.data.frame(Selection,
                             View,
                             Channel,
                             MinFreq,
                             MaxFreq,start.time,end.time,
                             Detections,Probability)

          RavenSelectionTableDFTemp <-
            RavenSelectionTableDFTemp[, c(
              "Selection",
              "View",
              "Channel",
              "start.time",
              "end.time",
              "MinFreq",
              "MaxFreq",
              'Probability',"Detections"
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
              "Detections"
            )

          RavenSelectionTableDF <- rbind.data.frame(RavenSelectionTableDF,
                                                    RavenSelectionTableDFTemp)




        }

        unlink('data/Temp/WavFiles', recursive = TRUE)
        unlink('data/Temp/Images/Images', recursive = TRUE)

        dir.create('data/Temp/WavFiles')
        dir.create('data/Temp/Images/Images')

        Detected <- str_detect(toString(predictedResnet),'1')
        Detected <- ifelse(Detected =='TRUE', "Gibbons",'NoGibbons')
        RecorderID <- paste(BoxDrivePathShort[x], q, sep='_')
        TempRow <- cbind.data.frame(Detected,RecorderID)
        SoundFileDF <- rbind.data.frame(SoundFileDF,TempRow)
        write.csv(SoundFileDF,'data/Images_allSoundfileDF.csv',row.names = F)
      }

      if(nrow(RavenSelectionTableDF) > 0){
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



