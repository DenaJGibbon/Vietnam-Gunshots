# Vietnam Binary Model Training ---------------------------------------------------------
# Load necessary packages and functions
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,# FALSE means the features are frozen
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnamunbalanced_binary_unfrozen_TRUE_/performance_tables/',
                                                             model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
(PerformanceOutPutTrained$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrainedBalanced <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_binary_unfrozen_TRUE_/performance_tables/',
                                                             model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrainedBalanced$f1_plot
PerformanceOutPutTrainedBalanced$best_f1$F1
PerformanceOutPutTrainedBalanced$pr_plot
(PerformanceOutPutTrainedBalanced$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam_belize/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam_belize'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = FALSE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrainedBalanced <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_belize_binary_unfrozen_TRUE_/performance_tables/',
                                                                     model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrainedBalanced$f1_plot
PerformanceOutPutTrainedBalanced$best_f1$F1
PerformanceOutPutTrainedBalanced$pr_plot
(PerformanceOutPutTrainedBalanced$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))



# Unfreeze = TRUE  -----------------------------------------------------------------

# Vietnam Binary Model Training ---------------------------------------------------------
# Load necessary packages and functions
devtools::load_all("/Users/denaclink/Desktop/RStudioProjects/gibbonNetR")

# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnamunbalanced/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnamunbalanced'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,# FALSE means the features are frozen
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrained <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnamunbalanced_binary_unfrozen_TRUE_/performance_tables/',
                                                             model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrained$f1_plot
PerformanceOutPutTrained$best_f1$F1
PerformanceOutPutTrained$pr_plot
(PerformanceOutPutTrained$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrainedBalanced <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_binary_unfrozen_TRUE_/performance_tables/',
                                                                     model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrainedBalanced$f1_plot
PerformanceOutPutTrainedBalanced$best_f1$F1
PerformanceOutPutTrainedBalanced$pr_plot
(PerformanceOutPutTrainedBalanced$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))


# Location of spectrogram images for training
input.data.path <-  'data/imagesvietnam_belize/'

# Location of spectrogram images for testing
test_data_path <- 'data/testdatacombined/'

# Training data folder short
trainingfolder.short <- 'imagesvietnam_belize'

# Number of epochs to include
epoch.iterations <- c(1,2,3,4,5,20)

# Train the models specifying different architectures
gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='alexnet',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")


gibbonNetR::train_CNN_binary(input.data.path=input.data.path,
                             architecture ='vgg16',
                             save.model= TRUE,
                             learning_rate = 0.001,
                             test.data=test.data.path,
                             unfreeze.param = TRUE,
                             epoch.iterations=epoch.iterations,
                             early.stop = "yes",
                             output.base.path = "model_output_finaltest/",
                             trainingfolder=trainingfolder.short,
                             positive.class="gunshot",
                             negative.class="noise")

PerformanceOutPutTrainedBalanced <- gibbonNetR::get_best_performance(performancetables.dir='/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/model_output_finaltest/_imagesvietnam_belize_binary_unfrozen_TRUE_/performance_tables/',
                                                                     model.type = 'binary',class='gunshot',Thresh.val =0)

PerformanceOutPutTrainedBalanced$f1_plot
PerformanceOutPutTrainedBalanced$best_f1$F1
PerformanceOutPutTrainedBalanced$pr_plot
(PerformanceOutPutTrainedBalanced$pr_plot)+scale_color_manual(values=matlab::jet.colors(6))

