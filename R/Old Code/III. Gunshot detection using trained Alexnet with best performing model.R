# Prepare data ------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(stringr)
library(tuneR)
library(seewave)
#library(gibbonR)

# Note: This can be used to run over entire sound files

# Load pre-trained models
# Based on the top performing model for each data type
modelAlexnetGunshot <- luz_load("data/_output_unfrozen_FALSE_imagesvietnam_belize_/_imagesvietnam_belize_1_modelalexNet.pt")

# Set path to BoxDrive
WavInput <- '/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/'

BoxDrivePath <- list.files(WavInput,
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files(WavInput,
                           full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 4
hop.size <- 3
threshold <- 0.5
OutputFolder <- 'data/Detections/AlexNetbelize/'
dir.create(OutputFolder)
Outputdirselection <- 'data/DetectionSelections/AlexNetbelize/'
dir.create(Outputdirselection)

for(x in rev(1:length(BoxDrivePath)) ){ tryCatch({
  RavenSelectionTableDFAlexnet <- data.frame()
  
  start.time.detection <- Sys.time()
  print(paste(x, 'out of', length(BoxDrivePath)))
  TempWav <- readWave(BoxDrivePath[x])
  WavDur <- seewave::duration(TempWav)
  
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
    unlink('/Users/denaclink/Desktop/data/Temp/WavFiles', recursive = TRUE)
    unlink('/Users/denaclink/Desktop/data/Temp/Images/Images', recursive = TRUE)
    
    dir.create('/Users/denaclink/Desktop/data/Temp/WavFiles')
    dir.create('/Users/denaclink/Desktop/data/Temp/Images/Images')
    
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
      writeWave(short.sound.files[[d]],paste('/Users/denaclink/Desktop/data/Temp/WavFiles','/',
                                             BoxDrivePathShort[x],'_',start.time[d], '.wav', sep=''),
                extensible = F)
    }
    
    # Save images to a temp folder
    print(paste('Creating images',start.time[1],'start time clips'))
    
    for(e in 1:length(short.sound.files)){
      jpeg(paste('/Users/denaclink/Desktop/data/Temp/Images/Images','/', BoxDrivePathShort[x],'_',start.time[e],'.jpg',sep=''),res = 50)
      short.wav <- short.sound.files[[e]]
      
      seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0,2),fastdisp=TRUE,noisereduction=1)
      
      graphics.off()
    }
    
    # Predict using Alexnet ----------------------------------------------------
    print('Classifying images using Alexnet')
    
    test.input <- '/Users/denaclink/Desktop/data/Temp/Images/'
    
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
    test_dl <- dataloader(test_ds, batch_size =nfiles,shuffle = FALSE)
    
    # Predict using Alexnet
    AlexnetPred <- predict(modelAlexnetGunshot, test_dl)
    AlexnetProb <- torch_sigmoid(AlexnetPred)
    AlexnetProb <-  1- as_array(torch_tensor(AlexnetProb, device = 'cpu'))
    AlexnetClass <- ifelse((AlexnetProb) > threshold, 'gunshot', 'noise')
    
    # Calculate the probability associated with each class
    Probability <- AlexnetProb
    

    image.files <- list.files(file.path(test.input),recursive = T,
                              full.names = T)
    nslash <- str_count(image.files,'/')+1
    nslash <- nslash[1]
    image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
    image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]
    
    print('Saving output')
    Detections <-  which(Probability > threshold )
    
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
    
    print('Saving output')
    file.copy(image.files[DetectionIndices],
              to= paste(OutputFolder,
                        image.files.short[DetectionIndices],
                        '_',
                        round(Probability[DetectionIndices],2),
                        '_Alexnet_.jpg', sep=''))
    
    Detections <- image.files.short[DetectionIndices]
    
    
    if (length(Detections) > 0) {
      Selection <- seq(1, length(Detections))
      View <- rep('Spectrogram 1', length(Detections))
      Channel <- rep(1, length(Detections))
      MinFreq <- rep(100, length(Detections))
      MaxFreq <- rep(2000, length(Detections))
      start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=3)[,3])
      end.time.new <- start.time.new + clip.duration
      Probability <- round(Probability[DetectionIndices],2)
      
      RavenSelectionTableDFAlexnetTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)
      
      RavenSelectionTableDFAlexnetTemp <-
        RavenSelectionTableDFAlexnetTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]
      
      colnames(RavenSelectionTableDFAlexnetTemp) <-
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
      
      RavenSelectionTableDFAlexnet <- rbind.data.frame(RavenSelectionTableDFAlexnet,
                                                     RavenSelectionTableDFAlexnetTemp)
      
   
      if(nrow(RavenSelectionTableDFAlexnet) > 0){
        csv.file.name <-
          paste(Outputdirselection,
                BoxDrivePathShort[x],
                'GunshotAlexnetAllFiles.txt',
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
      
      
    }
  }
  
  
  rm(TempWav)
  rm(short.sound.files)
  rm( test_ds )
  rm(short.wav)
  end.time.detection <- Sys.time()
  print(end.time.detection-start.time.detection)
}, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
}




# AlexNetbalanced Model -------------------------------------------------------------


# Load pre-trained models

modelAlexNetbalancedGunshot <- luz_load("data/_output_unfrozen_TRUE_imagesvietnam_/_imagesvietnam_1_modelAlexNet.pt")

# Set path to BoxDrive
WavInput <- '/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/'

BoxDrivePath <- list.files(WavInput,
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files(WavInput,
                                full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 4
hop.size <- 3
OutputFolder <- 'data/Detections/AlexNetimagesvietnam/'
dir.create(OutputFolder)
Outputdirselection <- 'data/DetectionSelections/AlexNetimagesvietnam/'
dir.create(Outputdirselection)

for(x in rev(1:length(BoxDrivePath)) ){ tryCatch({
  RavenSelectionTableDFAlexNetbalanced <- data.frame()
  
  start.time.detection <- Sys.time()
  print(paste(x, 'out of', length(BoxDrivePath)))
  TempWav <- readWave(BoxDrivePath[x])
  WavDur <- seewave::duration(TempWav)
  
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
    unlink('/Users/denaclink/Desktop/data/Temp/WavFiles', recursive = TRUE)
    unlink('/Users/denaclink/Desktop/data/Temp/Images/Images', recursive = TRUE)
    
    dir.create('/Users/denaclink/Desktop/data/Temp/WavFiles')
    dir.create('/Users/denaclink/Desktop/data/Temp/Images/Images')
    
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
      writeWave(short.sound.files[[d]],paste('/Users/denaclink/Desktop/data/Temp/WavFiles','/',
                                             BoxDrivePathShort[x],'_',start.time[d], '.wav', sep=''),
                extensible = F)
    }
    
    # Save images to a temp folder
    print(paste('Creating images',start.time[1],'start time clips'))
    
    for(e in 1:length(short.sound.files)){
      jpeg(paste('/Users/denaclink/Desktop/data/Temp/Images/Images','/', BoxDrivePathShort[x],'_',start.time[e],'.jpg',sep=''),res = 50)
      short.wav <- short.sound.files[[e]]
      
      seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0,2),fastdisp=TRUE,noisereduction=1)
      
      graphics.off()
    }
    
    # Predict using AlexNetbalanced ----------------------------------------------------
    print('Classifying images using AlexNetbalanced')
    
    test.input <- '/Users/denaclink/Desktop/data/Temp/Images/'
    
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
    test_dl <- dataloader(test_ds, batch_size =nfiles, shuffle = F)
    
    # Predict using AlexNetbalanced
    AlexNetbalancedPred <- predict(modelAlexNetbalancedGunshot, test_dl)
    AlexNetbalancedProb <- torch_sigmoid(AlexNetbalancedPred)
    AlexNetbalancedProb <-  1- as_array(torch_tensor(AlexNetbalancedProb, device = 'cpu'))
    AlexNetbalancedClass <- ifelse((AlexNetbalancedProb) > threshold, 'gunshot', 'noise')
    
    # Calculate the probability associated with each class
    Probability <- AlexNetbalancedProb
    
    OutputFolder <- 'data/Detections/AlexNetbalanced/'
    
    image.files <- list.files(file.path(test.input),recursive = T,
                              full.names = T)
    nslash <- str_count(image.files,'/')+1
    nslash <- nslash[1]
    image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
    image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]
    
    print('Saving output')
    Detections <-  which(Probability > threshold )
    
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
    
    print('Saving output')
    file.copy(image.files[DetectionIndices],
              to= paste(OutputFolder,
                        image.files.short[DetectionIndices],
                        '_',
                        round(Probability[DetectionIndices],2),
                        '_AlexNetbalanced_.jpg', sep=''))
    
    Detections <- image.files.short[DetectionIndices]
    
    
    if (length(Detections) > 0) {
      Selection <- seq(1, length(Detections))
      View <- rep('Spectrogram 1', length(Detections))
      Channel <- rep(1, length(Detections))
      MinFreq <- rep(100, length(Detections))
      MaxFreq <- rep(2000, length(Detections))
      start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=3)[,3])
      end.time.new <- start.time.new + clip.duration
      Probability <- round(Probability[DetectionIndices],2)
      
      RavenSelectionTableDFAlexNetbalancedTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)
      
      RavenSelectionTableDFAlexNetbalancedTemp <-
        RavenSelectionTableDFAlexNetbalancedTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]
      
      colnames(RavenSelectionTableDFAlexNetbalancedTemp) <-
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
      
      RavenSelectionTableDFAlexNetbalanced <- rbind.data.frame(RavenSelectionTableDFAlexNetbalanced,
                                                       RavenSelectionTableDFAlexNetbalancedTemp)
      
      if(nrow(RavenSelectionTableDFAlexNetbalanced) > 0){
        csv.file.name <-
          paste(Outputdirselection,
                BoxDrivePathShort[x],
                'GunshotAlexNetbalancedAllFiles.txt',
                sep = '')
        
        write.table(
          x = RavenSelectionTableDFAlexNetbalanced,
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
  
  
  rm(TempWav)
  rm(short.sound.files)
  rm( test_ds )
  rm(short.wav)
  end.time.detection <- Sys.time()
  print(end.time.detection-start.time.detection)
}, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
}


# AlexNetunbalanced Model -------------------------------------------------------------


# Load pre-trained models

modelAlexNetunbalancedGunshot <- luz_load("data/_output_unfrozen_FALSE_imagesvietnamunbalanced_/_imagesvietnamunbalanced_4_modelAlexNet.pt")

# Set path to BoxDrive
WavInput <- '/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/'

BoxDrivePath <- list.files(WavInput,
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files(WavInput,
                                full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 4
hop.size <- 3
OutputFolder <- 'data/Detections/AlexNetimagesvietnamunbalanced/'
dir.create(OutputFolder)
Outputdirselection <- 'data/DetectionSelections/AlexNetimagesvietnamunbalanced/'
dir.create(Outputdirselection)

for(x in rev(1:length(BoxDrivePath)) ){ tryCatch({
  RavenSelectionTableDFAlexNetunbalanced <- data.frame()
  
  start.time.detection <- Sys.time()
  print(paste(x, 'out of', length(BoxDrivePath)))
  TempWav <- readWave(BoxDrivePath[x])
  WavDur <- seewave::duration(TempWav)
  
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
    unlink('/Users/denaclink/Desktop/data/Temp/WavFiles', recursive = TRUE)
    unlink('/Users/denaclink/Desktop/data/Temp/Images/Images', recursive = TRUE)
    
    dir.create('/Users/denaclink/Desktop/data/Temp/WavFiles')
    dir.create('/Users/denaclink/Desktop/data/Temp/Images/Images')
    
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
      writeWave(short.sound.files[[d]],paste('/Users/denaclink/Desktop/data/Temp/WavFiles','/',
                                             BoxDrivePathShort[x],'_',start.time[d], '.wav', sep=''),
                extensible = F)
    }
    
    # Save images to a temp folder
    print(paste('Creating images',start.time[1],'start time clips'))
    
    for(e in 1:length(short.sound.files)){
      jpeg(paste('/Users/denaclink/Desktop/data/Temp/Images/Images','/', BoxDrivePathShort[x],'_',start.time[e],'.jpg',sep=''),res = 50)
      short.wav <- short.sound.files[[e]]
      
      seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,scale=F,grid=F,flim=c(0,2),fastdisp=TRUE,noisereduction=1)
      
      graphics.off()
    }
    
    # Predict using AlexNetunbalanced ----------------------------------------------------
    print('Classifying images using AlexNetunbalanced')
    
    test.input <- '/Users/denaclink/Desktop/data/Temp/Images/'
    
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
    test_dl <- dataloader(test_ds, batch_size =nfiles, shuffle = F)
    
    # Predict using AlexNetunbalanced
    AlexNetunbalancedPred <- predict(modelAlexNetunbalancedGunshot, test_dl)
    AlexNetunbalancedProb <- torch_sigmoid(AlexNetunbalancedPred)
    AlexNetunbalancedProb <-  1- as_array(torch_tensor(AlexNetunbalancedProb, device = 'cpu'))
    AlexNetunbalancedClass <- ifelse((AlexNetunbalancedProb) > threshold, 'gunshot', 'noise')
    
    # Calculate the probability associated with each class
    Probability <- AlexNetunbalancedProb
    
    OutputFolder <- 'data/Detections/AlexNetunbalanced/'
    
    image.files <- list.files(file.path(test.input),recursive = T,
                              full.names = T)
    nslash <- str_count(image.files,'/')+1
    nslash <- nslash[1]
    image.files.short <- str_split_fixed(image.files,pattern = '/',n=nslash)[,nslash]
    image.files.short <- str_split_fixed(image.files.short,pattern = '.jpg',n=2)[,1]
    
    print('Saving output')
    Detections <-  which(Probability > threshold )
    
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
    
    print('Saving output')
    file.copy(image.files[DetectionIndices],
              to= paste(OutputFolder,
                        image.files.short[DetectionIndices],
                        '_',
                        round(Probability[DetectionIndices],2),
                        '_AlexNetunbalanced_.jpg', sep=''))
    
    Detections <- image.files.short[DetectionIndices]
    
    
    if (length(Detections) > 0) {
      Selection <- seq(1, length(Detections))
      View <- rep('Spectrogram 1', length(Detections))
      Channel <- rep(1, length(Detections))
      MinFreq <- rep(100, length(Detections))
      MaxFreq <- rep(2000, length(Detections))
      start.time.new <- as.numeric(str_split_fixed(Detections,pattern = '_',n=3)[,3])
      end.time.new <- start.time.new + clip.duration
      Probability <- round(Probability[DetectionIndices],2)
      
      RavenSelectionTableDFAlexNetunbalancedTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)
      
      RavenSelectionTableDFAlexNetunbalancedTemp <-
        RavenSelectionTableDFAlexNetunbalancedTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]
      
      colnames(RavenSelectionTableDFAlexNetunbalancedTemp) <-
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
      
      RavenSelectionTableDFAlexNetunbalanced <- rbind.data.frame(RavenSelectionTableDFAlexNetunbalanced,
                                                             RavenSelectionTableDFAlexNetunbalancedTemp)
      
      if(nrow(RavenSelectionTableDFAlexNetunbalanced) > 0){
        csv.file.name <-
          paste(Outputdirselection,
                BoxDrivePathShort[x],
                'GunshotAlexNetunbalancedAllFiles.txt',
                sep = '')
        
        write.table(
          x = RavenSelectionTableDFAlexNetunbalanced,
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
  
  
  rm(TempWav)
  rm(short.sound.files)
  rm( test_ds )
  rm(short.wav)
  end.time.detection <- Sys.time()
  print(end.time.detection-start.time.detection)
}, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
}



