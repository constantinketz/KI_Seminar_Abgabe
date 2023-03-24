## Gruppenabgabe von Malte Neumann, Alexandra Weigel, Lukas Kleinert, Constantin Ketz, Moritz Kenk, Romy Glück, 
# Daniel Junginger, Timo Rahel

library(dplyr)
library(stringr)
library(plotly)
library(onewaytests)
library(car)

performance <- read.csv2(choose.files(), header = TRUE, sep= ",", dec=".", stringsAsFactors = T)
str(performance)
summary(performance)


#Correlation

cor(performance[c("math.score", "reading.score", "writing.score")])

mod1 <- lm(math.score ~ ., data = performance)


#Evaluation des Models

summary(mod1)

#plot

plot(x=predict(mod1), y= performance$math.score,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Linear Regression')
abline(a=0, b=1)


# fit eine neue lineare regression
mod2 <- lm(math.score ~ gender + race.ethnicity + lunch + test.preparation.course + reading.score + writing.score,data = performance)
summary(mod2)

plot(x=predict(mod2), y= performance$math.score,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Linear Regression')
abline(a=0, b=1)

###LASSO REGRESSION ###
library(glmnet)

y <- performance$math.score
x <- data.matrix(performance[,c("gender","race.ethnicity", "parental.level.of.education", "lunch", "test.preparation.course", "reading.score", "writing.score")])
cv_mod <- cv.glmnet(x,y,alpha=1)
best_lambda <- cv_mod$lambda.min
best_lambda

plot(cv_mod)

best_mod <- glmnet(x, y, alpha = 1, lambda = best_lambda)

coef(best_mod)

#best model for prediciton
y_predicted <- predict(best_mod, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq


plot(x=predict(best_mod, s = best_lambda, newx = x), y= performance$math.score,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Lasso Rgression')
abline(a=0, b=1)

### Ridge regression ###

y <- performance$math.score
x <- data.matrix(performance[, c("gender","race.ethnicity", "parental.level.of.education", "lunch", "test.preparation.course", "reading.score", "writing.score")])

#perform k-fold cross-validation to find optimal lambda value
cv_modR <- cv.glmnet(x, y, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambdaR <- cv_modR$lambda.min
best_lambdaR

#produce plot of test MSE by lambda value
plot(cv_modR)


best_modelR <- glmnet(x, y, alpha = 0, lambda = best_lambdaR)

coef(best_modelR)

#use fitted best model to make predictions
y_predicted <- predict(best_modelR, s = best_lambdaR, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq


plot(x=predict(best_modelR, s = best_lambdaR, newx = x), y= performance$math.score,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values Ridge Regression')
abline(a=0, b=1)
