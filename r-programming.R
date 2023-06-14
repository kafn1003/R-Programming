## Importing and Loading Libraries
library(tree)

## Opening/Reading the Dataset
# Heart Disease Prediction by PUSPITA SAHA
# https://www.kaggle.com/datasets/puspitasaha/heart-disease-prediction

heart_data <- read.csv("heart (1).csv")

## Data Exploration
dim(heart_data)

str(heart_data)

colnames(heart_data)

head(heart_data)

View(heart_data)

## Check for missing values
missing <- is.na(heart_data)

sum(missing)

head(missing)

## Building the Model using Decision Tree
heart_tree <- tree(as.factor(target)~., data = heart_data)

summary(heart_tree)

## Plotting the tree
plot(heart_tree)

text(heart_tree, pretty=0)

## Separate training and testing datasets
set.seed (40)

train <- sample(1: nrow( heart_data ), 665)

heart_tree2 <- tree(as.factor(target)~., heart_data, subset = train)

summary(heart_tree2)

# Plotting the tree after separating train and test 
plot(heart_tree2)

text(heart_tree2, pretty=0)

## Predicting the results using the test set
heart_pred <- predict(heart_tree2, heart_data[-train,], type="class")
heart_pred

## Creating a confusion matrix
cm <- with(heart_data[-train,], table(heart_pred, target))
cm

## Model Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
print(paste('Accuracy:', accuracy))

## Cross Validation
heart_cv <- cv.tree(heart_tree2, FUN = prune.misclass)
heart_cv

# Plotting the Cross Validation Model
plot(heart_cv)

## Pruning model
heart_prune <- prune.misclass(heart_tree2, best = 23)
heart_prune

# Plotting the Pruned Tree
plot(heart_prune)

text(heart_prune, pretty=0)

## Predicting the results AGAIN using the test set AFTER pruning
heart_pred2 <- predict(heart_prune, heart_data[-train,], type="class")
heart_pred2

cm2 <- with(heart_data[-train,], table(heart_pred2, target))
cm2

## Model Accuracy after Pruning
accuracy2 <- sum(diag(cm2)) / sum(cm2)
print(paste('Accuracy:', accuracy2))