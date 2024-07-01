# Load necessary libraries
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(glmnet)) install.packages("glmnet", dependencies = TRUE)
if (!require(pROC)) install.packages("pROC", dependencies = TRUE)
if (!require(rpart)) install.packages("rpart", dependencies = TRUE)
if (!require(rpart.plot)) install.packages("rpart.plot", dependencies = TRUE)
if (!require(mlbench)) install.packages("mlbench", dependencies = TRUE)
if (!require(randomForest)) install.packages("randomForest", dependencies = TRUE)

library(caret)
library(glmnet)
library(pROC)
library(rpart)
library(rpart.plot)
library(mlbench)
library(randomForest)


# Load the dataset
data(BreastCancer, package = "mlbench")
df <- BreastCancer

# Data preprocessing
df <- df[ , -1]  # Remove the ID column
df <- na.omit(df)  # Remove rows with missing values

# Convert factors to numeric
df$Class <- as.factor(ifelse(df$Class == "malignant", 1, 0))

# Separate features and target variable
X <- as.data.frame(sapply(df[ , -ncol(df)], as.numeric))
y <- df$Class

# Scale the features
preProcessParams <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preProcessParams, X)

# Split the data into training and testing sets (80% train, 20% test)
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[trainIndex, ]
X_test <- X_scaled[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Check lengths of training and test sets
cat("Length of X_train:", nrow(X_train), "\n")
cat("Length of y_train:", length(y_train), "\n")
cat("Length of X_test:", nrow(X_test), "\n")
cat("Length of y_test:", length(y_test), "\n")

# Feature selection using recursive feature elimination
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_results <- rfe(X_train, y_train, sizes = c(1:ncol(X_train)), rfeControl = control)
X_train <- X_train[, predictors(rfe_results)]
X_test <- X_test[, predictors(rfe_results)]

# Ensure y_train factor levels are valid R variable names
levels(y_train) <- make.names(levels(y_train))

# Train logistic regression model with cross-validation for hyperparameter tuning
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
tune_grid <- expand.grid(alpha = 0:1, lambda = seq(0.0001, 0.1, length = 20))

logistic_regression <- train(x = X_train, y = y_train,
                             method = "glmnet",
                             trControl = train_control,
                             tuneGrid = tune_grid,
                             metric = "ROC",
                             family = "binomial")


# Predict the test set results
y_pred_probs <- predict(logistic_regression, newdata = X_test, type = "prob")[,2]
y_pred <- ifelse(y_pred_probs > 0.5, 1, 0)

# Ensure y_pred and y_test are factors with the same levels
y_pred_factor <- factor(y_pred, levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

# Check the length of y_pred and y_test
cat("Length of y_pred:", length(y_pred), "\n")
cat("Length of y_test:", length(y_test), "\n")

# Ensure y_pred and y_test have the same length
if (length(y_pred) != length(y_test)) {
  stop("y_pred and y_test do not have the same length.")
}

# Calculate confusion matrix
conf_matrix <- confusionMatrix(y_pred_factor, y_test_factor)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# ROC curve
roc_curve <- roc(y_test_factor, as.numeric(y_pred_probs))
plot(roc_curve, col = "blue", main = "ROC Curve for Logistic Regression")
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

# Decision Tree Analysis
decision_tree <- rpart(Class ~ ., data = df[trainIndex,], method = "class")
rpart.plot(decision_tree)

# Predict with decision tree
y_pred_tree <- predict(decision_tree, newdata = df[-trainIndex,], type = "class")
y_pred_tree_factor <- factor(y_pred_tree, levels = c(0, 1))

# Confusion matrix for decision tree
conf_matrix_tree <- confusionMatrix(y_pred_tree_factor, y_test_factor)
print("Confusion Matrix for Decision Tree:")
print(conf_matrix_tree)
