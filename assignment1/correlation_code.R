setwd("/Users/oliviereverts/Documents/GitHub/FDS48/assignment1")
library(dplyr)
library(tiddle)
library(tidyr)

# Read in the ipums proportions CSV file
ipums_proportions <- read.csv("IPUMS_proportions.csv")

# Read in the twitter sentiment proportions 
twitter_proportions <- read.csv("sentiment_polarity_states_23k.csv")

# Merge the two datasets
merged_proportions <- merge(ipums_proportions, twitter_proportions, by.x = "State", by.y = "state")
View(merged_proportions)

# Make a subset with only the proportions 
sub_merged_proportions <- select(merged_proportions, Proportion.Some.College, Proportion.No.College, 
                                 Proportion.Not.In.Labor.Force, Proportion.Employed, Proportion.Unemployed,
                                 trump_ratio, clinton_ratio, total_ratio)

colnames(sub_merged_proportions) <- c("Individuals with Some College", "Individuals with No College", "Individuals not in Labor Force",
                                      "Individuals Employed", "Individuals Unemployed", "Trump Positive Tweet Ratio", "Clinton Negative 
                                      Tweet Ratio", "Total Positive Tweet Ratio")

View(sub_merged_proportions)

cor_mat <- cor(sub_merged_proportions)
cor_mat <- round(cor_mat, 2)
write.table(cor_mat, file="correlation_matrix.txt")