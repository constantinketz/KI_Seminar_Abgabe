# Lineare Regression




### Importieren aller relevanten Bibliotheken ###

library(dplyr)
library(stringr)
library(plotly)
library(onewaytests)
library(car)

#lade die Daten

insurance <- read.csv2("insurance.csv", header = TRUE, sep= ",", dec=".", stringsAsFactors = T)
str(insurance)
summary(insurance)


# berechne Correlation

cor(insurance[c("age", "bmi", "children", "charges")])



# fit eine lineare regression

model <- lm(charges ~ ., data = insurance)


#Evaluation des Models

summary(model)

#plot

plot(x=predict(model), y= insurance$charges,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Linear Regression')
abline(a=0, b=1)


# fit eine neue lineare regression
model2 <- lm(charges ~ bmi + age + smoker + children,data = insurance)
summary(model2)


###############################################
# fit eine neue lasso regression
library(glmnet)

#define response variable
y <- insurance$charges

#define matrix of predictor variables
x <- data.matrix(insurance[, c('age', 'sex', 'bmi', 'children', 'smoker', 'region')])

#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model) 


best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)

coef(best_model)

#use fitted best model to make predictions
y_predicted <- predict(best_model, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq


plot(x=predict(best_model, s = best_lambda, newx = x), y= insurance$charges,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Lasso Rgression')
abline(a=0, b=1)



###############################################
# fit eine neue ridge regression
library(glmnet)

#define response variable
y <- insurance$charges

#define matrix of predictor variables
x <- data.matrix(insurance[, c('age', 'sex', 'bmi', 'children', 'smoker', 'region')])

#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model) 


best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)

coef(best_model)

#use fitted best model to make predictions
y_predicted <- predict(best_model, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq


plot(x=predict(best_model, s = best_lambda, newx = x), y= insurance$charges,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Ridge Regression')
abline(a=0, b=1)

###############################################
# fit eine neue regression mit ANN
library(MASS)
library(neuralnet)

insurance$sex = str_replace_all(insurance$sex,c("female"="1", "male"="2"))
insurance$sex= as.numeric(insurance$sex)

insurance$smoker = str_replace_all(insurance$smoker,c("yes"="1", "no"="2"))
insurance$smoker= as.numeric(insurance$smoker)

insurance$region = str_replace_all(insurance$region,c("northeast"="1", "northwest"="2", "southeast"="3", "southwest"="4"))
insurance$region= as.numeric(insurance$region)

# Normalize the data
maxs <- apply(insurance, 2, max) 
mins <- apply(insurance, 2, min)
scaled <- as.data.frame(scale(insurance, center = mins, 
                              scale = maxs - mins))

# Split the data into training and testing set
index <- sample(1:nrow(insurance), round(0.75 * nrow(insurance)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

nn <- neuralnet(charges~age + sex + bmi + children + smoker + region, data = train_, hidden = c(5, 3), 
                linear.output = TRUE)

plot(nn)


# Predict on test data
pr.nn <- compute(nn, test_[,1:6])

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(insurance$charges) - min(insurance$charges)) 
+ min(insurance$charges)
test.r <- (test_$charges) * (max(insurance$charges) - min(insurance$charges)) + 
  min(insurance$charges)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot regression line
plot(test_$charges, pr.nn_, col = "red", 
     main = 'Real vs Predicted' Regression with ANN)

