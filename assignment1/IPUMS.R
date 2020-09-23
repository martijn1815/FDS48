install.packages("ipumsr")
library(ipumsr)

setwd("/Users/oliviereverts/Desktop/School:Work/UVA/Fundamentals of Data Science/Assignment 1")

ipums_ddi <- read_ipums_ddi("usa_00004.xml")
ipums_data <- read_ipums_micro(ipums_ddi)

head(ipums_data)

ipums_dataframe <- data.frame("State" = ipums_data$STATEFIP, "Educational Attainment" = ipums_data$EDUC, "Employment Status" = ipums_data$EMPSTAT)

write.csv(ipums_dataframe, "ipums_assignment1.csv")

