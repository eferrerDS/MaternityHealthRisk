#----Install libraries----
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(corrplot)   
library(e1071)
library(class)
library(randomForest)
library(dplyr)
library(ggplot2)
library(data.table)
library(readr)
library(caret)
library(knitr)


#----Importing dataset----
#Dataset file: "Maternal Health Risk Data Set.csv" included in files
#URL:https://archive.ics.uci.edu/dataset/863/maternal+health+risk

options(timeout = 120)




ds <- read_csv("Maternal Health Risk Data Set.csv")

#Training set and testing set----
set.seed(1, sample.kind = "Rounding") #if using R 3.6 or later
#set.seed(1) #if using R3.5 or earlier
final_test_index <- createDataPartition(y = ds$RiskLevel, times = 1, p = .1, list = FALSE)
temp_ds <- ds[-final_test_index,]
validation_index <- createDataPartition(y= temp_ds$RiskLevel, times = 1, p = .2, list = FALSE)

training <- temp_ds[-validation_index,]
validation <- temp_ds[validation_index,]
final_test <- ds[final_test_index,]

#Exploratory Analysis----
####Summary Excluding 'RiskLevel' column ----
summary(training[,-7])

#Age distribution by RiskLevel
training%>%ggplot(aes(Age, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Systolic distribution by RiskLevel
training%>%ggplot(aes(SystolicBP, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Diastolic distribution by RiskLevel
training%>%ggplot(aes(DiastolicBP, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Systolic - Diastolic by RiskLevel
training%>%ggplot(aes(SystolicBP-DiastolicBP, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

# training %>% ggplot(aes(SystolicBP, DiastolicBP, col = RiskLevel))+
#   geom_point(stat = "count", aes(size = Age))

#BS distribution by RiskLevel
training%>%ggplot(aes(BS, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Body temp distribution by RiskLevel
training%>%ggplot(aes(BodyTemp, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Heart rate distribution by RiskLevel
training%>%ggplot(aes(HeartRate, fill = RiskLevel))+
  geom_density()+
  theme_minimal()+
  scale_fill_brewer(palette="Set1") +
  facet_wrap(~RiskLevel)

#Calculate correlation Matrix and create a heatmap with correlation values
correlation_matrix <- cor(training[,c("Age","SystolicBP","DiastolicBP", "BS", "BodyTemp","HeartRate")])
# heatmap(correlation_matrix,
#         main = "Correlation Matrix Heatmap",
#         add.expr = {
#           corr_text <- round(correlation_matrix, 2)
#           text(expand.grid(1:ncol(correlation_matrix), 1:nrow(correlation_matrix)),
#                labels = as.vector(corr_text), cex = .7, col = "black")
#         })
#other correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", addCoef.col = "black", tl.col = "black")
#Methods----
##Naive Bayes----

# Create a Naive Bayes model
nb_model <- naiveBayes(RiskLevel ~ BodyTemp + HeartRate + DiastolicBP + SystolicBP + Age + BS,
                       data = training)

# Display the summary of the Naive Bayes model
print(nb_model)

# Make predictions on the training data
predictions <- predict(nb_model, validation)

# Display the confusion matrix
conf_matrix <- table(predictions, validation$RiskLevel)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")
results <- tibble(method = "Naive Bayes", Accuracy =sum(diag(conf_matrix)) / sum(conf_matrix))
##Random Forest----

# Copy datasets and convert "RiskLevel" to a factor
rf_training <- training
rf_training$RiskLevel <- as.factor(rf_training$RiskLevel)
rf_validation <- validation
rf_validation$RiskLevel <- as.factor(rf_validation$RiskLevel)

# Create a Random Forest model
rf_model <- randomForest(RiskLevel ~ BodyTemp + HeartRate + DiastolicBP + SystolicBP + Age + BS,
                         data = rf_training,
                         ntree = 500,  # Number of trees in the forest
                         importance = TRUE)

# Display the Random Forest model
print(rf_model)

# Make predictions on the training data
predictions <- predict(rf_model, newdata = rf_validation)

# Display the confusion matrix
conf_matrix <- table(predictions, rf_validation$RiskLevel)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")
results <- bind_rows(results, tibble(method = "Random Forest", Accuracy = sum(diag(conf_matrix)) / sum(conf_matrix)))

##K-nearest method (KNN)----

#Since we will need "RiskLevel" as a factor but we did it in the previous method we will use those datasets instead of creating new ones


# Create a k-NN model
knn_model <- knn(train = rf_training[, c("BodyTemp", "HeartRate", "DiastolicBP", "SystolicBP", "Age", "BS")],
                 test = rf_validation[, c("BodyTemp", "HeartRate", "DiastolicBP", "SystolicBP", "Age", "BS")],
                 cl = rf_training$RiskLevel,
                 k = 3)  # Specify the number of neighbors (k)

# Display the confusion matrix
conf_matrix <- table(knn_model, rf_validation$RiskLevel)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")
cat("Accuracy:", accuracy, "\n")
results <- bind_rows(results, tibble(method = "KNN", Accuracy = sum(diag(conf_matrix)) / sum(conf_matrix)))
# Since the best result was with random forest method, we will test the accuracy with our final_test dataset to get the final result
# Let's prepare the final_test data set by conferting RiskLevel to a factor
final_test$RiskLevel <- as.factor(final_test$RiskLevel)
# Make predictions on the final_test data

predictions <- predict(rf_model, newdata = final_test)

# Display the confusion matrix
conf_matrix <- table(predictions, final_test$RiskLevel)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")
results <- bind_rows(results, tibble(method = "Final", Accuracy = sum(diag(conf_matrix)) / sum(conf_matrix)))
results