# Create test files
# Prepare data ------------------------------------------------------------
library(luz)
library(torch)
library(torchvision)
library(torchdatasets)

library(stringr)
library(tuneR)
library(seewave)
library(gibbonR)


# Set path to BoxDrive
BoxDrivePath <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                           full.names = T, pattern='wav')

BoxDrivePathShort <- list.files('/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis/TestWavs',
                                full.names = F, pattern='wav')

BoxDrivePathShort <- str_split_fixed(BoxDrivePathShort,pattern = '.wav',n=2)[,1]

clip.duration <- 6
hop.size <- 3

for(x in 1:length(BoxDrivePath)){ tryCatch({

  TempWav <- readWave(BoxDrivePath[x])
  WavName <-  BoxDrivePathShort[x]
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

  lapply(1:length(short.sound.files),
                 function(i)
                   writeWave(
                     short.sound.files[[i]],
                     filename = paste('data/clips/test/',WavName,TempClips$ClipStart[i],'.wav',sep='_'),
                     extensible = F
                   ))

}, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") })
}


# Create spectrogram images -----------------------------------------------

SoundFiles <- list.files('data/clips/testaddedgunshot', recursive = T,full.names = T)
SoundFilesShort <- list.files('data/clips/testaddedgunshot', recursive = T,full.names = F)
Folder <- str_split_fixed(SoundFilesShort,pattern = '/',n=2)[,1]

SoundFilesShort <-str_split_fixed(SoundFilesShort,pattern = '/',n=2)[,2]
SoundFilesShort <- str_split_fixed(SoundFilesShort,pattern='.wav',n=2)[,1]



OutputFolder <- 'data/images/finaltestadded/'

for(y in 1:600){
  print(paste('processing',y, 'out of',length(SoundFiles) ))
  wav.rm <- str_split_fixed(SoundFilesShort[y],pattern='.wav',n=2)[,1]
  subset.directory <- paste(OutputFolder,Folder[y],sep='')
  jpeg(paste(subset.directory, '/', wav.rm,'.jpg',sep=''),res = 50)
  temp.name <- SoundFiles[y]
  short.wav <-readWave(temp.name)
  seewave::spectro(short.wav,tlab='',flab='',axisX=F,axisY = F,flim=c(0,2),scale=F,grid=F,noisereduction=1)
  graphics.off()

}



