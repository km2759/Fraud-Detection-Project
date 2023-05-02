library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(DMwR) # for smote implementation
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model


#Import data
df <- read.csv("creditcard.csv", header = TRUE)


#Remove 'Time' variable
df <- df[,-1]

#Change 'Class' variable to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")

#Scale numeric variables

df[,-30] <- scale(df[,-30])

head(df)
set.seed(123)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <-  subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# class ratio initially
table(train$Class)


# downsampling
set.seed(9560)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)

# upsampling
set.seed(9560)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

# smote
set.seed(9560)
smote_train <- SMOTE(Class ~ ., data  = train)

table(smote_train$Class)

# rose
set.seed(9560)
rose_train <- ROSE(Class ~ ., data  = train)$data 

table(rose_train$Class)


#Decision Trees
#CART Model Performance on imbalanced data
set.seed(5627)

orig_fit <- rpart(Class ~ ., data = train)


#Evaluate model performance on test set
pred_orig <- predict(orig_fit, newdata = test, method = "class")

# AUC on original data
roc.curve(test$Class, pred_orig[,2], plotit = TRUE)


set.seed(5627)
# Build down-sampled model

down_fit <- rpart(Class ~ ., data = down_train)

set.seed(5627)
# Build up-sampled model

up_fit <- rpart(Class ~ ., data = up_train)

set.seed(5627)
# Build smote model

smote_fit <- rpart(Class ~ ., data = smote_train)

set.seed(5627)
# Build rose model

rose_fit <- rpart(Class ~ ., data = rose_train)


# AUC on down-sampled data
pred_down <- predict(down_fit, newdata = test)

print('Fitting model to downsampled data')
roc.curve(test$Class, pred_down[,2], plotit = FALSE)

# AUC on up-sampled data
pred_up <- predict(up_fit, newdata = test)

print('Fitting model to upsampled data')
roc.curve(test$Class, pred_up[,2], plotit = FALSE)

# AUC on up-sampled data
pred_smote <- predict(smote_fit, newdata = test)

print('Fitting model to smote data')
roc.curve(test$Class, pred_smote[,2], plotit = FALSE)

# AUC on up-sampled data
pred_rose <- predict(rose_fit, newdata = test)

print('Fitting model to rose data')
roc.curve(test$Class, pred_rose[,2], plotit = FALSE)

#Logistic Regression
glm_fit <- glm(Class ~ ., data = up_train, family = 'binomial')

pred_glm <- predict(glm_fit, newdata = test, type = 'response')

roc.curve(test$Class, pred_glm, plotit = TRUE)


#Random Forest
x = up_train[, -30]
y = up_train[,30]

rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)


rf_pred <- predict(rf_fit, test[,-30], ctgCensus = "prob")
prob <- rf_pred$prob

roc.curve(test$Class, prob[,2], plotit = TRUE)

#XGBoost
# Convert class labels from factor to numeric

labels <- up_train$Class

y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)

set.seed(42)
xgb <- xgboost(data = data.matrix(up_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)
xgb_pred <- predict(xgb, data.matrix(test[,-30]))

roc.curve(test$Class, xgb_pred, plotit = TRUE)


#We can also take a look at the important features here.
names <- dimnames(data.matrix(up_train[,-30]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])



