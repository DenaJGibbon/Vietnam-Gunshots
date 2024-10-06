setwd("/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots")
# A. Vietnam Binary Model Training (unbalanced) ---------------------------------------------------------
# Load necessary packages and functions
library(gibbonNetR)

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Number of epochs to include
epoch.iterations <- c(1, 2, 3, 4, 5, 20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)

for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.25,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest_3/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}


# B. Vietnam Binary Model Training (balanced)  ---------------------------------------------------------
# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.5,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest_2/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}

# C. Vietnam Binary Model Training (plus Belize) ---------------------------------------------------------

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam_belize/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/test/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam_belize'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
architectures <- c('alexnet', 'vgg16', 'resnet18')
freeze.param <- c(TRUE, FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.5,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_finaltest_2/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}


# Evaluate performance of all models to report ----------------------------
# Reminder that par$requires_grad_(TRUE) means that it is unfrozen; e.g. there is fine-tuning
BasePath <- "/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest_2/"

performancetables.dir.multi.balanced.true <- paste(BasePath,'_imagesvietnamunbalanced_binary_unfrozen_TRUE_/performance_tables',sep='')
PerformanceOutputmulti.balanced.true <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.balanced.true,
                                                                           class='gunshot',
                                                                           model.type = "binary",
                                                                           Thresh.val = 0.1)

PerformanceOutputmulti.balanced.true$f1_plot
PerformanceOutputmulti.balanced.true$best_f1$F1
PerformanceOutputmulti.balanced.true$best_auc$AUC

performancetables.dir.multi.balanced.false <-paste(BasePath,'_imagesvietnamunbalanced_binary_unfrozen_FALSE_/performance_tables',sep='')
PerformanceOutputmulti.balanced.false <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.balanced.false,
                                                                            class='gunshot',
                                                                            model.type = "binary",
                                                                            Thresh.val = 0.1)

PerformanceOutputmulti.balanced.false$f1_plot
PerformanceOutputmulti.balanced.false$pr_plot
PerformanceOutputmulti.balanced.false$best_f1$F1
PerformanceOutputmulti.balanced.false$best_auc$AUC

performancetables.dir.multi.unbalanced.true <- paste(BasePath,'_imagesvietnam_binary_unfrozen_TRUE_/performance_tables',sep='')
PerformanceOutputmulti.unbalanced.true <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.unbalanced.true,
                                                           class='gunshot',
                                                           model.type = "binary",
                                                           Thresh.val = 0.1)

PerformanceOutputmulti.unbalanced.true$f1_plot
PerformanceOutputmulti.unbalanced.true$best_f1$F1
PerformanceOutputmulti.unbalanced.true$best_auc$AUC

performancetables.dir.multi.unbalanced.false <- paste(BasePath,'_imagesvietnam_binary_unfrozen_FALSE_/performance_tables',sep='')
PerformanceOutputmulti.unbalanced.false <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.unbalanced.false,
                                                                           class='gunshot',
                                                                           model.type = "binary",
                                                                           Thresh.val = 0.1)

PerformanceOutputmulti.unbalanced.false$f1_plot
PerformanceOutputmulti.unbalanced.false$best_f1$F1
PerformanceOutputmulti.unbalanced.false$best_auc$AUC

performancetables.dir.multi.addbelize.true <- paste(BasePath,'_imagesvietnam_belize_binary_unfrozen_TRUE_/performance_tables',sep='')
PerformanceOutputmulti.addbelize.true <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.addbelize.true,
                                                                      class='gunshot',
                                                                      model.type = "binary",
                                                                      Thresh.val = 0.1)

PerformanceOutputmulti.addbelize.true$f1_plot
PerformanceOutputmulti.addbelize.true$best_f1$F1
PerformanceOutputmulti.addbelize.true$best_auc$AUC

performancetables.dir.multi.addbelize.false <- paste(BasePath,'_imagesvietnam_belize_binary_unfrozen_FALSE_/performance_tables',sep='')
PerformanceOutputmulti.addbelize.false <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.addbelize.false,
                                                                          class='gunshot',
                                                                          model.type = "binary",
                                                                          Thresh.val = 0.1)

PerformanceOutputmulti.addbelize.false$f1_plot
PerformanceOutputmulti.addbelize.false$best_f1$F1
PerformanceOutputmulti.addbelize.false$best_auc$AUC

CombinedDF <- rbind.data.frame(PerformanceOutputmulti.balanced.true$best_f1,PerformanceOutputmulti.balanced.false$best_f1,
                 PerformanceOutputmulti.unbalanced.true$best_f1,PerformanceOutputmulti.unbalanced.false$best_f1,
                 PerformanceOutputmulti.addbelize.true$best_f1,PerformanceOutputmulti.addbelize.false$best_f1,
                 PerformanceOutputmulti.balanced.true$best_precision,PerformanceOutputmulti.balanced.false$best_precision,
                 PerformanceOutputmulti.unbalanced.true$best_precision,PerformanceOutputmulti.unbalanced.false$best_precision,
                 PerformanceOutputmulti.addbelize.true$best_precision,PerformanceOutputmulti.addbelize.false$best_precision,
                 PerformanceOutputmulti.balanced.true$best_recall,PerformanceOutputmulti.balanced.false$best_recall,
                 PerformanceOutputmulti.unbalanced.true$best_recall,PerformanceOutputmulti.unbalanced.false$best_recall,
                 PerformanceOutputmulti.addbelize.true$best_recall,PerformanceOutputmulti.addbelize.false$best_recall)

CombinedBestPerforming <- CombinedDF[,c("Precision", "Recall", "F1",
                                                    "Training Data", "N epochs", "CNN Architecture", "Threshold", "Frozen")]

CombinedBestPerforming$Precision <- round(CombinedBestPerforming$Precision,2)
CombinedBestPerforming$F1 <- round(CombinedBestPerforming$F1,2)
#CombinedBestPerforming$AUC <- round(CombinedBestPerforming$AUC,2)

CombinedBestPerforming$Recall<- round(CombinedBestPerforming$Recall,2)


CombinedBestPerforming$Precision <- format(CombinedBestPerforming$Precision,nsmall = 2)
CombinedBestPerforming$F1 <- format(CombinedBestPerforming$F1,nsmall = 2)
#CombinedBestPerforming$AUC <- format(CombinedBestPerforming$AUC,nsmall = 2)

CombinedBestPerforming$Threshold <- format(CombinedBestPerforming$Threshold,nsmall = 2)

CombinedBestPerforming$Recall <- format(CombinedBestPerforming$Recall,nsmall = 2)

CombinedBestPerforming$`Training Data` <- as.factor(CombinedBestPerforming$`Training Data`)
levels(CombinedBestPerforming$`Training Data` )

CombinedBestPerforming$`Training Data` <- plyr::revalue(CombinedBestPerforming$`Training Data`,
                                                        c("belizegunshots" = "Belize only",
                                                          "imagesvietnam" = "Vietnam balanced",
                                                          "imagesvietnam_belize"= "Vietnam + Belize",
                                                          "imagesvietnamunbalanced" = "Vietnam unbalanced"))

CombinedBestPerforming$`CNN Architecture` <- as.factor(CombinedBestPerforming$`CNN Architecture`)

CombinedBestPerforming$`CNN Architecture` <- plyr::revalue(CombinedBestPerforming$`CNN Architecture`,
                                                           c("alexnet"  = "AlexNet" , "vgg16"  = "VGG16"))

CombinedBestPerformingFlex <- flextable::flextable(CombinedBestPerforming)
CombinedBestPerformingFlex

# Note that for the 'Frozen' column, YES= fine-tuned, NO= no fine-tuning
#flextable::save_as_docx(CombinedBestPerformingFlex, path='Table 2 Performance Summary Updated.docx')


# Test for generalizability -------------------------------------------------------------------------

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test_data_path <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/data/imagesbelize/train/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Number of epochs to include
epoch.iterations <- c(2)

# Train the models specifying different architectures
architectures <- c('alexnet')
freeze.param <- c(FALSE)
for (a in 1:length(architectures)) {
  for (b in 1:length(freeze.param)) {
    gibbonNetR::train_CNN_binary(
      input.data.path = input.data.path,
      noise.weight = 0.25,
      architecture = architectures[a],
      save.model = TRUE,
      learning_rate = 0.001,
      test.data = test_data_path,
      unfreeze.param = freeze.param[b],
      # FALSE means the features are frozen
      epoch.iterations = epoch.iterations,
      list.thresholds = seq(0, 1, .1),
      early.stop = "yes",
      output.base.path = "model_output_testonbelize_V2/",
      trainingfolder = trainingfolder.short,
      positive.class = "gunshot",
      negative.class = "noise"
    )
    
  }
}


performancetables.dir.multi.testbelize <- '/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_testonbelize_V2/_imagesvietnamunbalanced_binary_unfrozen_FALSE_/performance_tables'
PerformanceOutputmulti.testbelize <- gibbonNetR::get_best_performance(performancetables.dir=performancetables.dir.multi.testbelize,
                                                                           class='gunshot',
                                                                           model.type = "binary",
                                                                           Thresh.val = 0.1)

PerformanceOutputmulti.testbelize$f1_plot
PerformanceOutputmulti.testbelize$best_f1$F1
PerformanceOutputmulti.testbelize$best_auc$AUC
as.data.frame(PerformanceOutputmulti.testbelize$best_f1)
