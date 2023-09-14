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
modelAlexnetGunshot <- luz_load("data/_output_unfrozen_TRUE_imagesvietnam_belize_/_imagesvietnam_belize_1_modelalexNet.pt")

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
    
    OutputFolder <- 'data/Detections/Alexnetbelize/'
    
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
          paste('data/DetectionSelections/AlexNetbelize/',
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




# VGG16balanced Model -------------------------------------------------------------


# Load pre-trained models

modelVGG16balancedGunshot <- luz_load("data/_output_unfrozen_TRUE_imagesvietnam_/_imagesvietnam_5_modelVGG16.pt")

# Set path to BoxDrive
WavInput <- '/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/'

BoxDrivePath <- list.files(WavInput,
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files(WavInput,
                                full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 4
hop.size <- 3


for(x in rev(1:length(BoxDrivePath)) ){ tryCatch({
  RavenSelectionTableDFVGG16balanced <- data.frame()
  
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
    
    # Predict using VGG16balanced ----------------------------------------------------
    print('Classifying images using VGG16balanced')
    
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
    
    # Predict using VGG16balanced
    VGG16balancedPred <- predict(modelVGG16balancedGunshot, test_dl)
    VGG16balancedProb <- torch_sigmoid(VGG16balancedPred)
    VGG16balancedProb <-  1- as_array(torch_tensor(VGG16balancedProb, device = 'cpu'))
    VGG16balancedClass <- ifelse((VGG16balancedProb) > threshold, 'gunshot', 'noise')
    
    # Calculate the probability associated with each class
    Probability <- VGG16balancedProb
    
    OutputFolder <- 'data/Detections/VGG16balanced/'
    
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
                        '_VGG16balanced_.jpg', sep=''))
    
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
      
      RavenSelectionTableDFVGG16balancedTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)
      
      RavenSelectionTableDFVGG16balancedTemp <-
        RavenSelectionTableDFVGG16balancedTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]
      
      colnames(RavenSelectionTableDFVGG16balancedTemp) <-
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
      
      RavenSelectionTableDFVGG16balanced <- rbind.data.frame(RavenSelectionTableDFVGG16balanced,
                                                       RavenSelectionTableDFVGG16balancedTemp)
      
      if(nrow(RavenSelectionTableDFVGG16balanced) > 0){
        csv.file.name <-
          paste('data/DetectionSelections/VGG16balanced/',
                BoxDrivePathShort[x],
                'GunshotVGG16balancedAllFiles.txt',
                sep = '')
        
        write.table(
          x = RavenSelectionTableDFVGG16balanced,
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


# VGG16unbalanced Model -------------------------------------------------------------


# Load pre-trained models

modelVGG16unbalancedGunshot <- luz_load("data/_output_unfrozen_FALSE_imagesvietnamunbalanced_/_imagesvietnamunbalanced_1_modelVGG16.pt")

# Set path to BoxDrive
WavInput <- '/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/WavsFinalPerformance/'

BoxDrivePath <- list.files(WavInput,
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files(WavInput,
                                full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 4
hop.size <- 3


for(x in rev(1:length(BoxDrivePath)) ){ tryCatch({
  RavenSelectionTableDFVGG16unbalanced <- data.frame()
  
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
    
    # Predict using VGG16unbalanced ----------------------------------------------------
    print('Classifying images using VGG16unbalanced')
    
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
    
    # Predict using VGG16unbalanced
    VGG16unbalancedPred <- predict(modelVGG16unbalancedGunshot, test_dl)
    VGG16unbalancedProb <- torch_sigmoid(VGG16unbalancedPred)
    VGG16unbalancedProb <-  1- as_array(torch_tensor(VGG16unbalancedProb, device = 'cpu'))
    VGG16unbalancedClass <- ifelse((VGG16unbalancedProb) > threshold, 'gunshot', 'noise')
    
    # Calculate the probability associated with each class
    Probability <- VGG16unbalancedProb
    
    OutputFolder <- 'data/Detections/VGG16unbalanced/'
    
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
                        '_VGG16unbalanced_.jpg', sep=''))
    
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
      
      RavenSelectionTableDFVGG16unbalancedTemp <-
        cbind.data.frame(Selection,
                         View,
                         Channel,
                         MinFreq,
                         MaxFreq,start.time.new,end.time.new,Probability,
                         Detections)
      
      RavenSelectionTableDFVGG16unbalancedTemp <-
        RavenSelectionTableDFVGG16unbalancedTemp[, c(
          "Selection",
          "View",
          "Channel",
          "start.time.new",
          "end.time.new",
          "MinFreq",
          "MaxFreq",
          'Probability',"Detections"
        )]
      
      colnames(RavenSelectionTableDFVGG16unbalancedTemp) <-
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
      
      RavenSelectionTableDFVGG16unbalanced <- rbind.data.frame(RavenSelectionTableDFVGG16unbalanced,
                                                             RavenSelectionTableDFVGG16unbalancedTemp)
      
      if(nrow(RavenSelectionTableDFVGG16unbalanced) > 0){
        csv.file.name <-
          paste('data/DetectionSelections/VGG16unbalanced/',
                BoxDrivePathShort[x],
                'GunshotVGG16unbalancedAllFiles.txt',
                sep = '')
        
        write.table(
          x = RavenSelectionTableDFVGG16unbalanced,
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




