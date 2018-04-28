# Importing libraries

library(broom)
library(mice)
library(dplyr)
library(vtreat)
library(caret)
library(ggplot2)
library(mlr)
library(corrplot)
library(stringr)
library(e1071)
library(xgboost)
library(magrittr)

# setting the input path and reading the train and test dataset 
setwd("B:/utd/comp/machienlearningpractice#1")
train_data <- read.csv("train_indessa.csv", stringsAsFactors = FALSE, na.strings = c("", " ", "NA"))
test_data <- read.csv("test_indessa.csv", stringsAsFactors = FALSE, na.strings = c("", " ", "NA"))


str(train_data)
str(test_data)

# converting the dependant variable as a factor
train_data$loan_status <- as.factor(train_data$loan_status)

#Checking the proportion of loan defaulters to see if we have class imbalance problem
round(prop.table(table(train_data$loan_status))*100)

# ---------------------------
# ///////////////////////////
#
#    DATA CLEANNING
#
# //////////////////////////
# ---------------------------


#Extracting term value and converting to integer
train_data$term <- unlist(str_extract_all(string = train_data$term,pattern = "\\d+"))
train_data$term <- as.integer(train_data$term)

test_data$term <- unlist(str_extract_all(string = test_data$term,pattern = "\\d+"))
test_data$term <- as.integer(test_data$term)

#Cleaning emoplyment length variable and converting to integer
train_data$emp_length <- gsub("< 1 year", "0", train_data$emp_length)
train_data$emp_length <- gsub("n/a", "-1", train_data$emp_length)
train_data$emp_length<- unlist(str_extract_all(string = train_data$emp_length,pattern = "\\d+"))
train_data$emp_length <- as.integer(train_data$emp_length)

test_data$emp_length <- gsub("< 1 year", "0", test_data$emp_length)
test_data$emp_length <- gsub("n/a", "-1", test_data$emp_length)
test_data$emp_length<- unlist(str_extract_all(string = test_data$emp_length,pattern = "\\d+"))
test_data$emp_length <- as.integer(test_data$emp_length)

#Checking number of observation in each level
xtabs(~ application_type, train_data)
xtabs(~ pymnt_plan, train_data)

#Dropping ID variables and variables which provide no value 
train_data['zip_code'] <- NULL
test_data['zip_code'] <- NULL

train_data['addr_state'] <- NULL
test_data['addr_state'] <- NULL

train_data['desc'] <- NULL
test_data['desc'] <- NULL

train_data['title'] <- NULL
test_data['title'] <- NULL

train_data['emp_title'] <- NULL
test_data['emp_title'] <- NULL

train_data['batch_enrolled'] <- NULL
test_data['batch_enrolled'] <- NULL

train_data['member_id'] <- NULL
test_data['member_id'] <- NULL

# Dropping some more variables as it mostly consists of one level and hence adds no value to predicting target variable
train_data['pymnt_plan'] <- NULL
test_data['pymnt_plan'] <- NULL

train_data['verification_status_joint'] <- NULL
test_data['verification_status_joint'] <- NULL

train_data['application_type'] <- NULL
test_data['application_type'] <- NULL


#Converting initial status, payment plan and application type to categorical vairable
train_data$initial_list_status[train_data$initial_list_status == "f"] <- 1
train_data$initial_list_status[train_data$initial_list_status == "w"] <- 0

test_data$initial_list_status[test_data$initial_list_status == "f"] <- 1
test_data$initial_list_status[test_data$initial_list_status == "w"] <- 0

train_data$initial_list_status <- as.factor(train_data$initial_list_status)
test_data$initial_list_status <- as.factor(test_data$initial_list_status)


#Finding out proportion of missing values in all the columns
missing_prop <- lapply(train_data, function(x) as.integer(prop.table(table(is.na(x)))*100))
str(missing_prop)

# Deleting all the columns with greater than 60% missing values.
train_data$mths_since_last_major_derog <- NULL
train_data$mths_since_last_record <- NULL

test_data$mths_since_last_major_derog <- NULL
test_data$mths_since_last_record <- NULL

#Checking for correlation
az <- split(names(train_data), sapply(train_data, function(x){ class(x)}))
#creating a data frame of numeric variables
xs <- train_data[c(az$numeric, az$integer)]
#check correlation

corrplot::corrplot(cor(xs, use = "pairwise.complete.obs"))
rm(xs)

#Removing variabels with high correlation
train_data[,c('collection_recovery_fee', 'funded_amnt_inv', 'loan_amnt')] <- NULL
test_data[,c('collection_recovery_fee', 'funded_amnt_inv', 'loan_amnt')] <- NULL


#Imputing rest of missing values with -1
missing_columns <- colnames(train_data)[colSums(is.na(train_data)) > 0]
for(i in 1:length(missing_columns)){
  train_data[[missing_columns[i]]][is.na(train_data[missing_columns[i]])] <- -1
}

missing_columns <- colnames(test_data)[colSums(is.na(test_data)) > 0]
for(i in 1:length(missing_columns)){
  test_data[[missing_columns[i]]][is.na(test_data[missing_columns[i]])] <- -1
}


#Checking for skewness and log transforming variables with skew greater than 2. 
az <- split(names(train_data), sapply(train_data, function(x){ class(x)}))
train_data[az$integer] <- lapply(train_data[az$integer], as.numeric)
test_data[az$integer] <- lapply(test_data[az$integer], as.numeric)

az <- split(names(train_data), sapply(train_data, function(x){ class(x)}))
skew <- sapply(train_data[az$numeric], function(x) skewness(x, na.rm = T))
skew <- skew[skew > 2]
train_data[,names(skew)] <- lapply(train_data[,names(skew)], function(x) log(x + 10))


skew <- sapply(test_data[az$numeric], function(x) skewness(x, na.rm = T))
skew <- skew[skew > 2]
test_data[,names(skew)] <- lapply(test_data[,names(skew)], function(x) log(x + 10))


#Cleaning few other variables
train_data$home_ownership <- as.factor(train_data$home_ownership)
train_data$home_ownership <- combineLevels(train_data$home_ownership, levs = c("OTHER", "ANY", "NONE"), newLabel = "OTHER")

test_data$home_ownership <- as.factor(test_data$home_ownership)
test_data$home_ownership <- combineLevels(test_data$home_ownership, levs = c("OTHER", "NONE"), newLabel = "OTHER")


train_data$last_week_pay[train_data$last_week_pay == "NAth week"] <- "-1"
train_data$last_week_pay <- unlist(str_extract_all(string = train_data$last_week_pay, pattern = "\\-*\\d+"))
train_data$last_week_pay <- as.integer(train_data$last_week_pay)

test_data$last_week_pay[test_data$last_week_pay == "NAth week"] <- "-1"
test_data$last_week_pay <- unlist(str_extract_all(string = test_data$last_week_pay, pattern = "\\-*\\d+"))
test_data$last_week_pay <- as.integer(test_data$last_week_pay)

# splitting up the character variables 
df_train <- train_data
df_test <- test_data
az <- split(names(df_train), sapply(df_train, function(x){ class(x)}))
df_train[az$character] <- lapply(df_train[az$character], as.factor)
df_test[az$character] <- lapply(df_test[az$character], as.factor)


# ---------------------------
# ///////////////////////////
#
#   FEATURE ENGINEERING
#
# //////////////////////////
# ---------------------------


df_train$income_loan_ratio <- df_train$annual_inc/ df_train$funded_amnt
df_train$interest <- df_train$funded_amnt*df_train$int_rate*df_train$term/(100*12)
df_train$unpaid_interest <- (df_train$interest - df_train$total_rec_int)/df_train$interest

df_test$income_loan_ratio <- df_test$annual_inc/ df_test$funded_amnt
df_test$interest <- df_test$funded_amnt*df_test$int_rate*df_test$term/(100*12)
df_test$unpaid_interest <- (df_test$interest - df_test$total_rec_int)/df_test$interest

#Removing variables which are not to be used further to free memory
rm(skew, missing_columns, missing_prop, az)

#checking importance of variables
library(randomForest)
library(e1071)

random.model = randomForest(loan_status ~., data = df_train, importance = TRUE, ntree = 100)

importance(random.model)
varImpPlot(random.model)

#Dropping variables which are unimportant
df_train$recoveries <- NULL
df_train$acc_now_delinq <- NULL
df_train$pub_rec <- NULL
df_train$collections_12_mths_ex_med <- NULL

df_test$recoveries <- NULL
df_test$acc_now_delinq <- NULL
df_test$pub_rec <- NULL
df_test$collections_12_mths_ex_med <- NULL

x <- df_train[, -c('loan_status')]
drop <- c("loan_status")
x <- df_train[, !names(df_train) %in% drop]
y <- as.factor(df_train$loan_status)


# ---------------------------
# ///////////////////////////
#
#     MODEL BUILDING
#
# //////////////////////////
# ---------------------------


#BUILDING CROSSVALIDATED RANDOM FOREST MODEL
#Using caret package

control <- trainControl(method = "repeatedcv", number = 5, search = "random")
set.seed(12)
tunegrid <- expand.grid(.mtry = c(3:6))
rf_grid <- caret::train(x, y, method = "rf",tunegrid = tunegrid, trControl = control)

#Using randomForest package

set.seed(100)
bestmtry <- tuneRF(x, y, stepFactor = 1, improve = 0.005, ntreeTry = 100)
model1 <- randomForest(loan_status ~., data = df_train, mtry = 5, ntree = 100)

predicted_value <- predict(model1, newdata = df_test)
submit <- data.frame(member_id = test_data$member_id, loan_status = predicted_value)
write.csv(submit, "sample_submission.csv",row.names = F)

#XGBOOST
#One Hot Encoding

vars <- colnames(x)
treatplan <- designTreatmentsZ(df_train, vars, verbose = FALSE)
newvars <- treatplan %>% 
  use_series(scoreFrame) %>%
  filter(code %in% c("clean", "lev")) %>%
  use_series(varName)

train.treat <- prepare(treatplan, df_train, varRestriction = newvars)
test.treat <- prepare(treatplan, df_test, varRestriction = newvars)

y <- as.numeric(y) - 1
cv <- xgb.cv(data = as.matrix(train.treat), label = y, nrounds = 100, nfold = 5, objective = "binary:logistic", eta = 0.3, max_depth = 6, early_stopping_rounds = 10, verbose = 0)
elog <- as.data.frame(cv$evaluation_log)
elog %>%
  summarize(ntree.train = which.min(train_error_mean), ntrees.test = which.min(test_error_mean)) 

#The best value for nrounds is 100
xgb_model <- xgboost(data = as.matrix(train.treat), # training data as matrix
                          label = y,  # column of outcomes
                          nrounds = 100,       # number of trees to build from cross validation
                          objective = "binary:logistic", # objective
                          eta = 0.3,
                          depth = 6,
                          verbose = 0  # silent
)

predictions <- predict(xgb_model, as.matrix(test.treat))

importance_matrix <- xgb.importance(colnames(train.treat), model = xgb_model)
xgb.plot.importance(importance_matrix[1:10,])

# creating a submission file
submit <- data.frame(member_id = test_data$member_id, loan_status = predictions)
write.csv(submit, "sample_submission2.csv",row.names = F)
