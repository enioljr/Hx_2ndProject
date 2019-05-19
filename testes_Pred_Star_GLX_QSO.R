#
#

#============================> TO BE USED / CHECKED
corr <- round(cor(sky.train[, -c(1, 10, 13, 14)], use = "complete.obs"), 2)

# Plot
install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("red1", "honeydew", "green2"), 
           title="Correlation of Numeric Features", 
           ggtheme=theme_bw)
#===============================>
#==== Now trying with the normalized set:

corr <- round(cor(sky.train.model[, -9], use = "complete.obs"), 2)

# Plot
install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("red1", "honeydew", "green2"), 
           title="Correlation of Numeric Features", 
           ggtheme=theme_bw)
#======================================= SAME THIGN = APPROVED!

#==================VARIABLE IMPORTANCE===================

mplot_importance <- function(var, imp, colours = NA, limit = 18, model_name = NA, subtitle = NA,
                             save = FALSE, file_name = "viz_importance.png", subdir = NA) {
  
  require(ggplot2)
  require(gridExtra)
  options(warn=-1)
  
  if (length(var) != length(imp)) {
    message("The variables and importance values vectors should be the same length.")
    stop(message(paste("Currently, there are",length(var),"variables and",length(imp),"importance values!")))
  }
  if (is.na(colours)) {
    colours <- "tomato3" 
  }
  out <- data.frame(var = var, imp = imp, Type = colours)
  if (length(var) < limit) {
    limit <- length(var)
  }
  
  output <- out[1:limit,]
  
  p <- ggplot(output, 
              aes(x = reorder(var, imp), y = imp * 100, 
                  label = round(100 * imp, 1))) + 
    geom_col(aes(fill = Type), width = 0.1) +
    geom_point(aes(colour = Type), size = 6) + 
    coord_flip() + xlab('') + theme_minimal() +
    ylab('Importance') + 
    geom_text(hjust = 0.5, size = 2, inherit.aes = TRUE, colour = "white") +
    labs(title = paste0("Variables Importances. (", limit, " / ", length(var), " plotted)"))
  
  if (length(unique(output$Type)) == 1) {
    p <- p + geom_col(fill = colours, width = 0.2) +
      geom_point(colour = colours, size = 6) + 
      guides(fill = FALSE, colour = FALSE) + 
      geom_text(hjust = 0.5, size = 2, inherit.aes = TRUE, colour = "white")
  }
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  }  
  if(save == TRUE) {
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir))
      file_name <- paste(subdir, file_name, sep="/")
    }
    p <- p + ggsave(file_name, width=7, height=6)
  }
  
  return(p)
  
}
mplot_importance(class, sky.train[, 1:18], colours = NA, limit = 18, model_name = sky.train, subtitle = NA,
                 save = FALSE, file_name = "viz_importance.png", subdir = NA)
#=========================END======================= NÃO FUNCIONOU ================
install.packages("corrplot")
library(corrplot)
corr <- cor(as.matrix(sky.train.norm[, -c(9, 16:18)]))
colnames(corr) <- c("ra", "dec", "u", "g", "r", "i", "z", "run", "camcol",
                        "field", "redshift", "plate", "mjd", "fiberid")
corrplot.mixed(corr, lower = "number", upper = "circle",
               tl.col="black", tl.pos = "lt")

# Seems as if we have some multi-collinearity between in our parameters.
high <- caret::findCorrelation(cor((sky.train[, -c(1, 10, 13, 14)])), cutoff = 0.9)
sky.train$class1 <- as.factor(ifelse(sky.train$class == 0, "No", "Yes"))

#===============================> ???

#================================
# Plotting the features
#================================
require("cowplot")
library(cowplot)
theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
                          legend.position="top")

theme2<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
                          legend.position="none")

plot_grid(ggplot(sky.train.norm, aes(ra, fill = class)) + geom_density(alpha = 0.4)+theme1, 
          ggplot(sky.train.norm, aes(dec, fill = class)) + geom_density(alpha = 0.4)+theme1,
          align = "h") # skies group

plot_grid(ggplot(sky.train.norm, aes(ra, fill = class)) + geom_density(alpha = 0.4)+theme1, 
          ggplot(sky.train.norm, aes(z, fill = class)) + geom_density(alpha = 0.4)+theme1,
          ggplot(sky.train.norm, aes(u, fill = class)) + geom_density(alpha = 0.4)+theme1,
          ggplot(sky.train.norm, aes(g, fill = class)) + geom_density(alpha = 0.4)+theme2,
          ggplot(sky.train.norm, aes(r, fill = class)) + geom_density(alpha = 0.4)+theme2,
          ggplot(sky.train.norm, aes(i, fill = class)) + geom_density(alpha = 0.4)+theme2,
          align = "h") # Thuann_Gunn group

plot_grid(ggplot(sky.train.norm, aes(run, fill = class)) + geom_density(alpha = 0.4)+theme1, 
          ggplot(sky.train.norm, aes(camcol, fill = class)) + geom_density(alpha = 0.4)+theme1,
          ggplot(sky.train.norm, aes(field, fill = class)) + geom_density(alpha = 0.4)+theme2,
          align = "h") # Field Features group

plot_grid(ggplot(sky.train.norm, aes(redshift, fill = class)) + geom_density(alpha = 0.4)+theme1, 
          ggplot(sky.train.norm, aes(plate, fill = class)) + geom_density(alpha = 0.4)+theme1,
          ggplot(sky.train.norm, aes(mjd, fill = class)) + geom_density(alpha = 0.4)+theme1,
          ggplot(sky.train.norm, aes(fiberid, fill = class)) + geom_density(alpha = 0.4)+theme2,
          align = "h") # Other Features group
#==========================>

# KNN

## normalizing data in the validation set:
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
validation.norm$class <- as.factor(validation.norm$class) # convert to factors
head(validation.norm, 2)
library(class)

set.seed(2205)
pred.knn <- knn(sky.train.norm[, -c(16:18)], validation.norm[, -c(16:18)],
            cl = sky.train.norm$class, k = 5)
knn.cm <- caret::confusionMatrix(pred.knn, validation.norm$class)
knn.cm$overall[["Accuracy"]] # good accuracy of 0.89

# HClustering

set.seed(2205)
pred.hc <- hclust(dist(sky.train.model[, -9]), method = 'complete')
clusters <- cutree(pred.hc, k= 3)
clustered.sky <- mutate(sky.train.model, cluster = clusters)
clustered.sky %>% group_by(cluster) %>%
  data.frame() %>%
  plot_grid(ggplot(aes(x = class, y = redshift, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            ggplot(aes(x = class, y = u, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            ggplot(aes(x = class, y = g, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            ggplot(aes(x = class, y = r, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            ggplot(aes(x = class, y = i, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            ggplot(aes(x = class, y = z, color = factor(cluster)))+
            geom_point(alpha = 0.5)+geom_jitter(),
            allign = "h")

#======================================>

NZV <- nearZeroVar(sky.train)
plot(NZV)

#======================================>
# KNN do curso Hx

control <- trainControl(method = "cv", number = 10, p = .9)
set.seed(2205)
train_knn <- caret::train(class ~ ., method = "knn", 
                      data = sky.train.model,
                      tuneGrid = data.frame(k = seq(1, 10, 1)),
                      trControl = control)
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
train_knn$finalModel
# Now we fit for the entire dataset with the chosen k =1
fit_knn <- caret::knn3(class ~ .,  data = sky.train.model, k = 1)
y_hat_knn <- predict(fit_knn, validation.model, type="class")
cm <- caret::confusionMatrix(y_hat_knn, validation.model$class)
cm$overall["Accuracy"]

#==================TESTANDO NO DATASET PURO ====================

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# 5.2 Build Models
sky.train.model2 <- sky.train %>%
  select(-c("ra", "dec", "run", "rerun", "camcol", "field", "fiberid",
            "objid", "specobjid"))
# a) linear algorithms
set.seed(2205)
fit.lda <- caret::train(class ~ . , data=sky.train.model2, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(2205)
fit.cart <- caret::train(class ~ . , data=sky.train.model2, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(2205)
fit.knn <- caret::train(class ~ . , data=sky.train.model2, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(2205)
fit.svm <- caret::train(class ~ . , data=sky.train.model2, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(2205)
fit.rf <- caret::train(class ~ . , data=sky.train.model2, method="rf", metric=metric, trControl=control)
# 5.3 Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
print(fit.rf)


# 6 Make Predictions
# estimate skill of RF on the validation dataset

## normalizing data in the validation set:
set.seed(2205)
# columns to be discarded:
validation.model2 <- validation %>%
  select(-c("ra", "dec", "run", "rerun", "camcol", "field", "fiberid",
            "objid", "specobjid"))

predictions <- predict(fit.rf, validation.model2)
caret::confusionMatrix(predictions, validation.model2$class)


# Confusion Matrix and Statistics

#Reference
#Prediction GALAXY QSO STAR
#GALAXY    993  12    1
#QSO         5 158    0
#STAR        1   0  829

#Overall Statistics

#Accuracy : 0.9905          
#==========================END======================

#===================================================
#
# Applying SMOTE in the pure dataset
# Later: SMOTE in the model dataset
#
#==================================================

# columns to be discarded:

sky.train.model <- sky.train.norm %>%
  select(-c("ra", "dec", "run", "rerun", "camcol", "field", "fiberid",
            "objid", "specobjid"))


########=========#######
##### SMOTE #######
#####============#######

library(DMwR)
sky.smote <- SMOTE(class ~ ., data = sky.train[, -c(1, 10, 13)], perc.over = 200, k = 5, perc.under = 300)
prop.table(table(sky.train$class))
prop.table(table(sky.smote$class))

# 5 Evaluate some algorithms

# 5.1 Testing Harness
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# 5.2 Build Models

# a) linear algorithms
set.seed(2205)
fit.lda <- caret::train(class ~ . , data=sky.smote, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(2205)
fit.cart <- caret::train(class ~ . , data=sky.smote, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(2205)
fit.knn <- caret::train(class ~ . , data=sky.smote, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(2205)
fit.svm <- caret::train(class ~ . , data=sky.smote, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(2205)
fit.rf <- caret::train(class ~ . , data=sky.smote, method="rf", metric=metric, trControl=control)
# 5.3 Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
print(fit.rf)


# 6 Make Predictions
# estimate skill of RF on the validation dataset

## normalizing data in the validation set:
set.seed(2205)
validation.smote <- SMOTE(class ~ ., data = validation[, -c(1, 10, 13)], perc.over = 200, k = 5, perc.under = 300)
prop.table(table(validation$class))
prop.table(table(validation.smote$class))


predictions <- predict(fit.rf, validation.smote)
caret::confusionMatrix(predictions, validation.smote$class)

# RESULTS

#Confusion Matrix and Statistics

#Reference
#Prediction GALAXY QSO STAR
#GALAXY    547  18    1
#QSO        10 492    0
#STAR        2   0  460

#Overall Statistics

#Accuracy : 0.9797

# -=-=-=-=-=-=-=-=-=--=-=-=-=-
# SMOTE in the model dataset
# -=-=-==--=-=-=-=-=-=-=-=--=

sky.train.model <- sky.train.norm %>%
  select(-c("ra", "dec", "run", "rerun", "camcol", "field", "fiberid",
            "objid", "specobjid"))
sky.smote.model <- SMOTE(class ~ ., data = sky.train.model, perc.over = 200, k = 5, perc.under = 300)
prop.table(table(sky.train.model$class))
prop.table(table(sky.smote.model$class))
# 5 Evaluate some algorithms

# 5.1 Testing Harness
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# 5.2 Build Models

# a) linear algorithms
set.seed(2205)
fit.lda <- caret::train(class ~ . , data=sky.smote.model, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(2205)
fit.cart <- caret::train(class ~ . , data=sky.smote.model, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(2205)
fit.knn <- caret::train(class ~ . , data=sky.smote.model, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(2205)
fit.svm <- caret::train(class ~ . , data=sky.smote.model, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(2205)
fit.rf <- caret::train(class ~ . , data=sky.smote.model, method="rf", metric=metric, trControl=control)
# 5.3 Select best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)
print(fit.rf)


# 6 Make Predictions
# estimate skill of RF on the validation dataset

## normalizing data in the validation set:
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
# columns to be discarded:
validation.model <- validation.norm %>%
  select(-c("ra", "dec", "run", "rerun", "camcol", "field", "fiberid",
            "objid", "specobjid"))

validation.smote.model <- SMOTE(class ~ ., data = validation.model, perc.over = 200, k = 5, perc.under = 300)
prop.table(table(validation.model$class))
prop.table(table(validation.smote.model$class))

predictions <- predict(fit.rf, validation.smote.model)
caret::confusionMatrix(predictions, validation.smote.model$class)

# RESULTS

# Confusion Matrix and Statistics

#Reference
#Prediction GALAXY QSO STAR
#GALAXY    507  43    0
#QSO         8 467    0
#STAR       44   0  461

#Overall Statistics

#Accuracy : 0.9379

#================ END ===============

# Novo teste com sky.train.model pro RF

set.seed(2205)
rf.sky.train.normX <- randomForest(class ~ ., data = sky.train.model)
imp.df <- importance(rf.sky.train.normX) # importance of the features
imp.df <- data.frame(features = row.names(imp.df), MDG = imp.df[,1])
imp.df <- imp.df[order(imp.df$MDG, decreasing = TRUE),]
ggplot(imp.df, aes(x = reorder(features, MDG), y = MDG, fill = MDG)) +
  geom_bar(stat = "identity") + labs (x = "Features", y = "Mean Decrease Gini (MDG)") +
  coord_flip()


####### ATENÇÃO #########

# abaixo as linhas que não funcionaram mas estavam funcionando
# linha 194

# We still have some class distribution problems.
percentage <- prop.table(table(sky.train.model$class)) * 100
cbind(freq=table(sky.train$class), percentage=percentage)

# AS discussed before, this could be solved using the SMOTE function.
set.seed(2205)
sky.smote <- SMOTE(class ~ ., data = sky.train.model, perc.over = 200, k = 5, perc.under = 300)
percentage <- prop.table(table(sky.smote$class)) * 100
cbind(freq=table(sky.smote$class), percentage=percentage)

# Our class imbalance has been solved using this method.
# So we will run our models with this method first and then
# re-run our models using other methods

# linha 


# The redshift

ggplot(sky.train, aes(plate, redshift, col = class)) + geom_point()

ggplot(sky.train, aes(redshift, plate, col = class)) + geom_point()

ggplot(sky.train.norm, aes(redshift, z, col = class)) + geom_point()

require("cowplot")
library(cowplot)
theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
               legend.position="top")

theme2<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
               legend.position="none")

plot_grid(ggplot(sky.train.norm, aes(redshift, z, col = class)) + geom_point() + theme1, 
          ggplot(sky.train.norm, aes(redshift, i, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(redshift, g, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(redshift, r, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(redshift, u, col = class)) + geom_point() + theme1,
          align = "h") # Thuann_Gunn group

plot_grid(ggplot(sky.train.norm, aes(z, redshift, col = class)) + geom_point() + theme1, 
          ggplot(sky.train.norm, aes(i, redshift, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(g, redshift, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(r, redshift, col = class)) + geom_point() + theme1,
          ggplot(sky.train.norm, aes(u, redshift, col = class)) + geom_point() + theme1,
          align = "h")
"dec", "run", "field", "ra", "z", "i", "g", "r", "u"