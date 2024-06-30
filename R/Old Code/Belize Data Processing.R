devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

TrainingDatapath <- '/Volumes/DJC Files 1/GunshotDataWavBelize/Training data'

gibbonNetR::spectrogram_images(
  trainingBasePath = TrainingDatapath,
  outputBasePath   = 'data/imagesbelize/',
  splits           = c(0.7, 0.3, 0),  # 70% training, 30% validation
  minfreq.khz = 0,
  maxfreq.khz = 2,
  new.sampleratehz= 'NA'
)

library(stringr)

# Match clips to smaller dataset

ImagesFile <- list.files('data/imagesbelize/',
                         recursive = T,full.names = T)

ImagesFileShort <- basename(ImagesFile)
ImagesFileShort <- str_split_fixed(ImagesFileShort,pattern = '.jpg', n=2)[,1]



WavFile <- list.files('/Volumes/DJC Files 1/GunshotDataWavBelize/Training data/Noise/',
                      recursive = T,full.names = T)

WavFileShort <- basename(WavFile)
WavFileShort <- str_split_fixed(WavFileShort,pattern = '.WAV', n=2)[,1]

OutputDir <- '/Volumes/DJC Files 1/GunshotDataWavBelize/Training data reduced/'

file.copy(
  from=WavFile[which((WavFileShort %in% ImagesFileShort))],
  to= paste(OutputDir,'noise/',WavFileShort[which((WavFileShort %in% ImagesFileShort))],'.wav',sep=''))


TestDatapath <- '/Volumes/DJC Files/GunshotDataWavBelize/Validation data/'

gibbonNetR::spectrogram_images(
  trainingBasePath = TestDatapath,
  outputBasePath   = 'data/imagesbelizetest/',
  splits           = c(0, 0, 1),  # 70% training, 30% validation
  minfreq.khz = 0,
  maxfreq.khz = 2,
  new.sampleratehz= 'NA'
)

# Evaluate on Belize data

# Location of spectrogram images for training
input.data.path <-  'data/imagesbelize/'

# Location of spectrogram images for testing
test.data.path <- 'data/imagesbelizetest/test/'

# Training data folder short
trainingfolder.short <- 'belizegunshots'

# Whether to unfreeze.param the layers
unfreeze.param <- TRUE # FALSE means the features are frozen; TRUE unfrozen

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Allow early stopping?
early.stop <- 'yes'

gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='resnet18',
                             noise.weight = 0.25,
                             learning_rate = 0.001,
                             save.model= TRUE,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             list.thresholds = seq(0.1, 1, .1),
                             output.base.path = "data/belizetest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="Gunshot",
                             negative.class="Noise")

performancetables.dir <- 'data/belizetest/_belizegunshots_binary_unfrozen_TRUE_/performance_tables/'

PerformanceOutput <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir,
                                                      class='Gunshot',
                                                      model.type = "binary",
                                                      Thresh.val = 0)
as.data.frame(PerformanceOutput$best_f1)
PerformanceOutput$f1_plot
PerformanceOutput$pr_plot
PerformanceOutput$FPRTPR_plot
PerformanceOutput$best_f1$F1
as.data.frame(PerformanceOutput$best_auc)

# Test on Vietnam data -------------------------------------------

trained_models_dir <- 'data/belizetest/_belizegunshots_binary_unfrozen_TRUE_/'


image_data_dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/imagesvietnam/test/'

evaluate_trainedmodel_performance(trained_models_dir=trained_models_dir,
                                  image_data_dir=image_data_dir,
                                  positive.class = 'gunshot',  #' Label for positive class
                                  negative.class = 'noise',    #' Label for negative class
                                  output_dir = "data/belizetest/")


PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='data/belizetest/performance_tables_trained/',
                                                             model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
PerformanceOutPutTrained$best_auc$AUC
(PerformanceOutPutTrained$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


# Belize model on Belize --------------------------------------------------
trained_models_dir <- 'data/belizetest/_belizegunshots_binary_unfrozen_TRUE_/'

image_data_dir <- 'data/imagesbelizetest/test/'

evaluate_trainedmodel_performance(trained_models_dir=trained_models_dir,
                                  image_data_dir=image_data_dir,
                                  positive.class = 'Gunshot',  #' Label for positive class
                                  negative.class = 'Noise',    #' Label for negative class
                                  output_dir = "data/belizetestonbelizedata/")


PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='data/belizetestonbelizedata/performance_tables_trained/',
                                                             model.type = 'binary',class='Gunshot',Thresh.val =0)

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
PerformanceOutPutTrained$best_auc$AUC
(PerformanceOutPutTrained$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))



# Vietnam model on Belize -------------------------------------------------
trained_models_dir <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output/_imagesvietnamunbalanced_binary_unfrozen_TRUE_/'

image_data_dir <- 'data/imagesbelizetest/test/'

evaluate_trainedmodel_performance(trained_models_dir=trained_models_dir,
                                  image_data_dir=image_data_dir,
                                  positive.class = 'Gunshot',  #' Label for positive class
                                  negative.class = 'Noise',    #' Label for negative class
                                  output_dir = "data/vietnammodelsonbelize/")


PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='data/vietnammodelsonbelize/performance_tables_trained/',
                                                             model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
(PerformanceOutPutTrained$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


