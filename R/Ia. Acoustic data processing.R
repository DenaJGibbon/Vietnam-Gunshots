#library(gibbonR)
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
for( a in 1: length(ListSelectionTables)){
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

# Check for missing soundfiles
ProcessedClips <- list.files('/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/clips',recursive = T)
ListMissingFiles <- list()

for(z in 1:length(ListWavFilesShort)){
  
  AlreadyProcessed <- which(str_detect( ProcessedClips,ListWavFilesShort[z]))
  print(AlreadyProcessed)
  if(length(AlreadyProcessed)==0){
    ListMissingFiles[[z]] <- ListWavFilesShort[z]
  }
}

ListMissingFiles


# Combine Selection Tables ------------------------------------------------

# List text files in directory
ListSelectionTables <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                  pattern = '.txt',full.names =T)

ListSelectionTablesShort <- list.files("/Users/denaclink/Library/CloudStorage/Box-Box/Gunshot analysis",
                                       pattern = '.txt',full.names =F)

CombinedDF <- data.frame()
# Combine in table
for( a in 1: length(ListSelectionTables)){
  print(paste('processing', a, 'out of',length(ListSelectionTables)))
  TempSelection <- read.delim( ListSelectionTables[a])
  TempSelection$Filename <- basename(ListSelectionTables[a])
  CombinedDF <- rbind.data.frame(CombinedDF,TempSelection )
}

nrow(CombinedDF)

write.csv(CombinedDF,'GunShotAnnotationsCombinedDF.csv',row.names = F)

