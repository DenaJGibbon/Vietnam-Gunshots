library(ggpubr)
library(tidyr)
library(cowplot)

SpatialRandom <- read.csv("/Users/denaclink/Desktop/RStudioProjects/Vietnam-Gunshots/Number of Gunshot_Detected_forRandomization.csv")

SpatialRandom <- na.omit(SpatialRandom)

# Plot historgram
Histogramofcounts <- gghistogram(data=SpatialRandom, x="UpdatedNumber" )+ ylab('Count') + xlab('Gunshots per site')

# How many gunshots?
TotalGunshots <- sum(SpatialRandom$UpdatedNumber)
print(TotalGunshots)

# Randomization 
# Function to reduce the total sum of UpdatedNumber by a specified amount
reduce.sum <- function(data, reduction.amount) {
  # Get the indices where UpdatedNumber is non-zero
  non.zero.indices <- which(data$UpdatedNumber != 0)
  
  if (length(non.zero.indices) == 0) {
    stop("No non-zero UpdatedNumber values found.")
  }
  
  # Calculate current sum
  current.sum <- sum(data$UpdatedNumber)
  
  # Check if reduction is possible
  if (current.sum < reduction.amount) {
    stop("Reduction amount is greater than the current total sum of UpdatedNumber.")
  }
  
  # Copy the data to modify
  modified.data <- data
  
  while (reduction.amount > 0 && length(non.zero.indices) > 0) {
    # Randomly select an index from non.zero.indices
    index <- sample(non.zero.indices, 1)
    
    # Determine the amount to reduce
    reduction <- min(modified.data$UpdatedNumber[index], reduction.amount)
    
    # Reduce the value
    modified.data$UpdatedNumber[index] <- modified.data$UpdatedNumber[index] - reduction
    
    # Decrease the reduction amount
    reduction.amount <- reduction.amount - reduction
    
    # the index if the value reaches zero
    if (modified.data$UpdatedNumber[index] == 0) {
      non.zero.indices <- non.zero.indices[non.zero.indices != index]
    }
  }
  
  return(modified.data)
}

RandomizationDF <- data.frame()
for( a in 1:25){
# Reduce the total sum by 50
result.50 <- reduce.sum(SpatialRandom, 50)
print("After reducing sum by 50:")
print(result.50)


# Reduce the total sum by 25
result.25 <- reduce.sum(SpatialRandom, 25)
print("After reducing sum by 25:")
print(result.25)


# Reduce the total sum by 50
result.10 <- reduce.sum(SpatialRandom, 10)
print("After reducing sum by 10:")
print(result.10)

# Reduce the total sum by 75
result.5 <- reduce.sum(SpatialRandom, 5)
print("After reducing sum by 5:")
print(result.5)


# If the test is significant the distributions are different
result.50p <- wilcox.test(result.50$UpdatedNumber,SpatialRandom$UpdatedNumber)$p.value
result.25p <- wilcox.test(result.25$UpdatedNumber,SpatialRandom$UpdatedNumber)$p.value
result.10p <- wilcox.test(result.10$UpdatedNumber,SpatialRandom$UpdatedNumber)$p.value
result.5p <- wilcox.test(result.5$UpdatedNumber,SpatialRandom$UpdatedNumber)$p.value
Index <- a
TempRow <- cbind.data.frame(result.50p,result.25p,result.10p,result.5p,Index)
colnames(TempRow) <- c('50','25','10','5','Index')
RandomizationDF <- rbind.data.frame(RandomizationDF,TempRow)
}


RandomizationDFlong <- gather(RandomizationDF, n.remove, p.value, '50':'5', factor_key=TRUE)

BoxplotRandom <- ggboxplot(data=RandomizationDFlong,x='n.remove', y='p.value')+ xlab('Number of gunshots removed')+ ylab('P-value')+
  geom_hline(yintercept =0.05, col = "red", lty='dashed')


cowplot::plot_grid(Histogramofcounts,BoxplotRandom, labels=c('A','B'),label_x =0.9)
