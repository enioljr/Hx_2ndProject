
#===============================================>
train <- sky.train[,-c(1,10,13)]
test <- validation[,-c(1,10,13)]

Model_poly <- ksvm(class~ ., data = train, scale = FALSE, kernel = "polydot")
Eval_poly <- predict(Model_poly, test[,-11])

#confusion matrix - Polynomial Kernel
caret::confusionMatrix(Eval_poly,test$class)

Model_RBF <- ksvm(class~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF <- predict(Model_RBF, test[,-11])

#confusion matrix - RBF Kernel
caret::confusionMatrix(Eval_RBF,test$class)

trainControl <- caret::trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(2205)
grid <- expand.grid(.sigma=c(0.01,0.02,0.03,0.04,0.05), .C=c(5,6,7,8) )

fit.svm <- caret::train(class~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl= trainControl)

print(fit.svm)


# Plotting "fit.svm" results
plot(fit.svm)

#Evaluating on the test data

evaluate_kernel_test<- predict(fit.svm, test[, -14])
confusionMatrix(evaluate_kernel_test, test$class)

#$#$#$#$#$#$#$

fiveMetric <- function(...) c(twoClassSummary(...),
                              defaultSummary(...))
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = fiveMetric,
                     classProbs = T,
                     verboseIter = T)
ctrlNoProb <- trainControl(method = "cv",
                           number = 10,
                           summaryFunction = twoClassSummary,
                           verboseIter = T)
# Random Forest Model #
set.seed(2205)
rf_model <- train(class ~ .,data = sky.model,
                  method = "rf",
                  ntree = 1500,
                  trControl = ctrl,
                  metric = "ROC")
plot(rf_model)

#=========================>>>

plot(fit.rf)
plot(varImp(fit.rf))

#==========================================
#
# testando com o norm dataset
#
#==========================================

sky.NORM <- sky.train.norm[,c("redshift", "z", "i", "g", "r", "u", "class")] # selecting the columns we need

# 5 Evaluate some algorithms

# 5.1 Testing Harness
# Run algorithms using 20-fold cross validation
control <- caret::trainControl(method="cv", number=20)
metric <- "Accuracy"

# 5.2 Build Models

# a) linear algorithms
set.seed(2205)
fit.lda <- caret::train(class ~ . , data=sky.NORM, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(2205)
fit.cart <- caret::train(class ~ . , data=sky.NORM, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(2205)
fit.knn <- caret::train(class ~ . , data=sky.NORM, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(2205)
fit.svm <- caret::train(class ~ . , data=sky.NORM, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(2205)
fit.rf <- caret::train(class ~ . , data=sky.NORM, method="rf", metric=metric, trControl=control)

# 5.3 Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm,
                          rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
print(fit.rf) # 0.9921259
plot(fit.rf)
plot(varImp(fit.rf))

# 6 Make Predictions
# estimate skill of RF on the validation dataset

# columns to be discarded:
set.seed(2205)
validation.norm <- normalizeData(validation[,-c(1, 13, 14)], type = "0_1") # using the package 'RSNNS'
validation.norm <- as.data.frame(validation.norm)
summary(validation.norm) # check the normalization

names_validation <- names(validation[,-c(1, 13, 14)]) # add the names non-numeric columns back to the df.
names(validation.norm) <- names_validation
head(validation.norm, 2) # now we check if it worked. That seems OK.
# now we add back the columns that were not included in the normalization.
validation.norm <- add_column(validation.norm, objid = validation$objid)
validation.norm <- add_column(validation.norm, specobjid = validation$specobjid)
validation.norm <- add_column(validation.norm, class = validation$class)
head(validation.norm, 2)
validation.NORM <- validation.norm[,c("redshift", "z", "i", "g", "r", "u", "class")]
set.seed(2205)
predictions <- predict(fit.rf, validation.NORM)
caret::confusionMatrix(predictions, validation.NORM$class) # 0.993


Y<-NULL
k<-30
set.seed(2205)
for (i in 1:k){
  fit.rf.cost <- caret::train(class ~ . , data=sky.model, method="rf", metric=metric,
                              trControl=control, cost = i)
  pred <- predict(fit.rf.cost, validation.model)
  tab <- table(as.numeric(validation.model$class), as.numeric(pred))
  Y[i] <- accuracy(tab)
}

Nbc=str_c("count=", as.character(which.max(Y)))

plot(x=1:k, y=Y, pch=20, main="Accuracy in function of cost", xlab="Cost", ylab="Accuracy in %")
abline(v=which.max(Y), lty=2,lwd=1)
text(x=which.max(Y)-2, y=min(Y),labels=Nbc)


#$%ˆ$%ˆ$%ˆ$%ˆ$%ˆ$%ˆ$%ˆ$ˆ%$%ˆ$ˆ%$%$ˆ%$%ˆ$%ˆ$ˆ%$%ˆ$ˆ%$$ˆ$ˆ$ˆˆ%######


# 1.2 Libraries used
library(tidyverse)
library(data.table)
library(caret)
library(ggplot2)
library(ggcorrplot)
library(RSNNS)
library(randomForest)
library(ggcorrplot)
library(kernlab)
library(cowplot)

# 2. Load The Data

list.files(path = ".") # list files on the current working directory
sky.df <- fread(file = "Skyserver_SQL2_27_2018 6_51_39 PM.csv", sep=",")

# 2.3. Create a Validation Dataset

# create a list of 80% of the rows in the original dataset we can use for training
index <- createDataPartition(sky.df$class, p=0.8, list=FALSE)
# select 20% of the data for validation
validation <- sky.df[-index,]
# use the remaining 80% of data to training and testing the models
sky.train <- sky.df[index,]

validation <- validation %>% 
  semi_join(sky.train, by = "class")

# 3. Summarize Dataset
str(sky.train)
summary(sky.train)
colSums(is.na(sky.train)) # any NA's?
dim(sky.train)

# 3.1 Types of attributes
sapply(sky.train, class)

# The "class" column is our response variable. Since this is a classification problem, we
# will transform it in a factor with three levels
sky.train$class <- as.factor(sky.train$class)
levels(sky.train$class)
validation$class <- as.factor(validation$class)
levels(validation$class)

# 3.2 Class distribution

# summarize the class distribution
percentage <- prop.table(table(sky.train$class)) * 100
cbind(freq=table(sky.train$class), percentage=percentage)


thuan_gunn <- c("u", "g", "r", "i", "z")

# field are features which describe a field within an image taken by the SDSS.
#A field is basically a part of the entire image corresponding to 2048 by 1489 pixels. 
field_feat <- c("run", "rerun", "camcol", "field")

# Right ascension (abbreviated RA) is the angular distance measured eastward along
# the celestial equator from the Sun at the March equinox to the hour circle of the
# point above the earth in question. When paired with declination (abbreviated dec),
# these astronomical coordinates specify the direction of a point on the celestial
# sphere (traditionally called in English the skies or the sky) in the equatorial
# coordinate system.
skies <- c("ra", "dec")

# remaining features
equipment_feat <- c("redshift", "plate", "mjd", "fiberid") # These features are related to the measurement equipment

# split input and output
x <- sky.train[,-c(1, 13, 14, 16, 17)]
y <- sky.train$class


ggplot(sky.train, aes(class, fill = class)) +
  geom_bar()

#  We need a different approach now: normalize the data to have all the numeric features
# between 0 and 1 to be able to compare them and understand better the data.
set.seed(2205)
sky.train.norm <- normalizeData(sky.train[,-c(1, 13, 14)], type = "0_1") # using the package 'RSNNS'
sky.train.norm <- as.data.frame(sky.train.norm)
summary(sky.train.norm) # check the normalization

names_sky.train <- names(sky.train[,-c(1, 13, 14)]) # add the names non-numeric columns back to the df.
names(sky.train.norm) <- names_sky.train
head(sky.train.norm, 2) # now we check if it worked. That seems OK.
# now we add back the columns that were not included in the normalization.
sky.train.norm <- add_column(sky.train.norm, objid = sky.train$objid)
sky.train.norm <- add_column(sky.train.norm, specobjid = sky.train$specobjid)
sky.train.norm <- add_column(sky.train.norm, class = sky.train$class)
head(sky.train.norm, 2)



# Let's run a model with randomForest to check the variables importance
set.seed(2205)
rf.sky.train <- randomForest(class ~ ., data = sky.train[, -c(1, 10, 13, 16, 17)])
imp.df <- importance(rf.sky.train) # importance of the features
imp.df <- data.frame(features = row.names(imp.df), MDG = imp.df[,1])
imp.df <- imp.df[order(imp.df$MDG, decreasing = TRUE),]
ggplot(imp.df, aes(x = reorder(features, MDG), y = MDG, fill = MDG)) +
  geom_bar(stat = "identity") + labs (x = "Features", y = "Mean Decrease Gini (MDG)") +
  coord_flip()



PCA.sky.train <- prcomp(sky.train[, -c(1, 10, 13, 14)])
summary(PCA.sky.train)
# We notice that the first 6 components respond by 99.99% of the data.
# Considering that the number of features is not so big, we can consider using all the features
# aleready considered for the PCA analysis or use the first 6 features.

rownames(imp.df) <- NULL
imp.df %>% knitr::kable(caption = "Importance")

# defining the dataset for the model:

sky.model <- sky.train[, -c(1, 10, 13)]
# We can also confirm the importance of the features in this correlation plot:
corr <- round(cor(sky.model[, -7], use = "complete.obs"), 2)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("red1", "honeydew", "green2"), 
           title="Correlation of Numeric Features", 
           ggtheme=theme_bw)
# Based on the PCA analysis, on the correlation of the numeric features, and on the MDG values
# we decided to select c("redshift", "z", "i", "g", "r", "u") as the most important
# features that explain 99.97% of the dataset.

sky.model <- sky.train[,c("redshift", "z", "i", "g", "r", "u", "class")] # selecting the columns we need


# 5 Evaluate some algorithms

# 5.1 Testing Harness
# Run algorithms using 20-fold cross validation
control <- caret::trainControl(method="cv", number=20)
metric <- "Accuracy"

# 5.2 Build Models

# a) linear algorithms
set.seed(2205)
fit.lda <- caret::train(class ~ . , data=sky.model, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(2205)
fit.cart <- caret::train(class ~ . , data=sky.model, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(2205)
fit.knn <- caret::train(class ~ . , data=sky.model, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(2205)
fit.svm <- caret::train(class ~ . , data=sky.model, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(2205)
fit.rf <- caret::train(class ~ . , data=sky.model, method="rf", metric=metric, trControl=control)

# 5.3 Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm,
                          rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
print(fit.rf) # 0.9915019
plot(fit.rf)
plot(varImp(fit.rf))

# 6 Make Predictions
# estimate skill of RF on the validation dataset

# columns to be discarded:
validation.model <- validation[,c("redshift", "z", "i", "g", "r", "u", "class")]

set.seed(2205)
predictions <- predict(fit.rf, validation.model)
caret::confusionMatrix(predictions, validation.model$class) # 0.992