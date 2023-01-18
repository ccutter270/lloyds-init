# NAMES: Caroline Cutter & Julia Fairbank

# CSCI 0311 - Artificial Intelligence 

# ASSIGNMENT: FINAL PROJECT
#             Lloyd's Algorithm with Different Initialization 

#DUE: Monday, May 23



# ----------------------------------- CODE -----------------------------------


# PACKAGES 

library("tidyverse")
library("dplyr")

setwd("~/Desktop/final")

data <- read_csv("lloyds_data.csv")                              # data file
data <- rename(data, "init" = "Initialization Method")





# ------------ GRAPHS ------------


# BOXPLOT - Accuracy
boxplot(data$Accuracy ~ data$init, main="Accuracy of Initializations",
        show.names=TRUE, 
        xlab="Initialization Method", ylab="Accuracy",
        col=(c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" )))


# BOXPLOT - Iterations 
boxplot(data$Iterations ~ data$init, main="Iterations of Initializations",
        show.names=TRUE, 
        xlab="Initialization Method", ylab="Iterations",
        col=(c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" )))


# BOXPLOT - Time 
boxplot(data$Time ~ data$init, main="Total Time of Initializations",
        show.names=TRUE, 
        xlab="Initialization Method", ylab="Time",
        col=(c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" )))





# AVERAGE OF DATA 
Accuracy_Summary <- data %>%
  group_by(init) %>%
  summarise_at(vars(Accuracy), list(accuracy = mean))

Iterations_Summary <- data %>%
  group_by(init) %>%
  summarise_at(vars(Iterations), list(Iterations = mean))

Time_Summary <- data %>%
  group_by(init) %>%
  summarise_at(vars(Time), list(time = mean))



# BAR GRAPHS

# Accuracy
ggplot(data=Accuracy_Summary, aes(x=init, y=accuracy, fill = init)) +
  geom_bar(stat="identity", width=0.5, color="black") +
  scale_fill_manual(values=c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" ))+
  labs(title = "Average Accuracy", x = "Initalization Method", y = "Accuracy (%)")




# Accuracy
ggplot(data=Iterations_Summary, aes(x=init, y=Iterations, fill = init)) +
  geom_bar(stat="identity", width=0.5, color="black") +
  scale_fill_manual(values=c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" ))+
  labs(title = "Average Iterations", x = "Initalization Method", y = "Iterations")



# Accuracy
ggplot(data=Time_Summary, aes(x=init, y=time, fill = init)) +
  scale_fill_manual(values=c("indianred1","tan1", "lightgreen", "lightblue", "mediumpurple1" ))+
  geom_bar(stat="identity", width=0.5, color="black") +
  labs(title = "Average Time", x = "Initalization Method", y = "Time (Seconds)")
























