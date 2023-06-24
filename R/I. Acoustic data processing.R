library(gibbonR)
library(tuneR)
library(seewave)
library(signal)
library(stringr)

# Create short .wav clips -------------------------------------------------

# List text files in directory
ListSelectionTables <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
           pattern = '.txt',full.names =T)

ListSelectionTablesShort <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                  pattern = '.txt',full.names =F)

# List .wav files in directory
ListWavFiles <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                  pattern = '.wav',full.names =T)

ListWavFilesShort <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                           pattern = '.wav',full.names =F)

ListWavFilesShort <- str_split_fixed(ListWavFilesShort,pattern = '.wav',n=2)[,1]

# Prepare clips
for( a in 40: length(ListSelectionTables)){
  print(paste('processing', a, 'out of',length(ListSelectionTables)))
  TempSelection <- read.delim( ListSelectionTables[a])
  TempWav <- readWave(ListWavFiles[a])
  shortSoundFile <- lapply(1:(nrow(TempSelection)),
                              function(i)
                                extractWave(
                                  TempWav,
                                  from = TempSelection$Begin.Time..s.[i]-1.5,
                                  to = TempSelection$End.Time..s.[i]+1.5,
                                  xunit = c("time"),
                                  plot = F,
                                  output = "Wave"
                                ))

  WavName <- ListWavFilesShort[a]

  lapply(1:(length(shortSoundFile)),
         function(i)
           writeWave(
             shortSoundFile[[i]],
             filename = paste('data/clips/gunshot/',WavName,i,'.wav',sep='_'),
             extensible = F
           ))

  # # Create noise clips for selections with >1 gunshots
  # if(nrow(TempSelection)>1){
  #
  #  Temp.Noise.Wavs <-  extractWave(
  #     TempWav,
  #     from = TempSelection$End.Time..s.[1],
  #     to = TempSelection$Begin.Time..s.[2],
  #     xunit = c("time"),
  #     plot = F,
  #     output = "Wave"
  #   )
  #
  #  TempSeq <- seq(1,duration(Temp.Noise.Wavs),3)
  #  length.seq <- length(TempSeq)-1
  #
  #  shortNoiseFile <- lapply(1:(length.seq),
  #                           function(i)
  #                             extractWave(
  #                               Temp.Noise.Wavs,
  #                               from = TempSeq[i],
  #                               to = TempSeq[i+1],
  #                               xunit = c("time"),
  #                               plot = F,
  #                               output = "Wave"
  #                             ))
  #
  #  lapply(1:length(shortNoiseFile),
  #         function(i)
  #           writeWave(
  #             shortNoiseFile[[i]],
  #             filename = paste('data/clips/noise/',WavName,i,'.wav',sep='_'),
  #             extensible = F
  #           ))
  # }
  #

}



# Create images for training ----------------------------------------------
# Image creation -----------------------------------------------
TrainingFolders <- list.files('data/clips',full.names = T)
TrainingFoldersShort <- list.files('data/clips',full.names = F)
OutputFolder <- 'data/images'
FolderVec <- c('train','valid','test') # Need to create these folders

for(z in 1:length(TrainingFolders)){
  SoundFiles <- list.files(TrainingFolders[z], recursive = T,full.names = T)
  SoundFilesShort <- list.files(TrainingFolders[z], recursive = T,full.names = F)

  for(y in 1:length(SoundFiles)){

    DataType <-  FolderVec[1]

     if (y%%5 == 0) {
       DataType <-  FolderVec[2]
     }

    if (y%%10 == 0) {
      DataType <-  FolderVec[3]
    }

    subset.directory <- paste(OutputFolder,DataType,TrainingFoldersShort[z],sep='/')

    if (!dir.exists(subset.directory)){
      dir.create(subset.directory)
      print(paste('Created output dir',subset.directory))
    } else {
      print(paste(subset.directory,'already exists'))
    }
    wav.rm <- str_split_fixed(SoundFilesShort[y],pattern='.wav',n=2)[,1]
    jpeg(paste(subset.directory, '/', wav.rm,'.jpg',sep=''),res = 50)
    temp.name <- SoundFiles[y]
    short.wav <-readWave(temp.name)
    seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,flim=c(0,2),scale=F,grid=F,noisereduction=1)
    graphics.off()

  }
}



