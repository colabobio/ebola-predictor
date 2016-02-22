# Fits the data using an Elastic Net with 50% mixture between L1 and L2 penalties.
# https://cran.r-project.org/web/packages/glmnet/index.html
# http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
# http://machinelearningmastery.com/penalized-regression-in-r/

args <- commandArgs(trailingOnly = TRUE)
nboot <- as.integer(args[1])

library(glmnet)

dat <- read.table("./models/test/training-data-completed.csv", sep=",", header=TRUE)
labels <- dat["OUT"]
y <- as.matrix(dat[,1])
x <- as.matrix(dat[,2:ncol(dat)])

# alpha=0 is Ridge Regression (L1 norm penalty only)
# alpha=0.5 is elastic net (mixture of L1 and L2 at a 50%)
# alpha=1 is lasso (L2 norm penalty only)
aelast=0.5

predfam="gaussian"

# Finds optimal lambda by cross-validation
cv <- cv.glmnet(x, y, family=predfam, alpha=aelast, nfolds=10)
lbest <- cv$lambda.min

fit <- glmnet(x, y, family=predfam, alpha=aelast, lambda=lbest)

library(ROCR)
prob <- predict(fit, x, type="link")
pred <- prediction(prob, dat["OUT"])
auc <- performance(pred, measure = "auc")
auc_app <- auc@y.values[[1]]

optim <- function(dat0, idx)
{
  x0 <- as.matrix(dat0[,2:ncol(dat0)])
  dat1 <- dat0[idx,]
    
  y1 <- as.matrix(dat1[,1])
  x1 <- as.matrix(dat1[,2:ncol(dat1)])  
  fit1 <- glmnet(x1, y1, family=predfam, alpha=aelast, lambda=lbest)
  
  prob1 <- predict(fit1, x1, type="link")
  prob0 <- predict(fit1, x0, type="link")
  pred1 <- prediction(prob1, dat1["OUT"])
  pred0 <- prediction(prob0, dat0["OUT"])  
  auc1 <- performance(pred1, measure = "auc")
  auc0 <- performance(pred0, measure = "auc")  
  auc1@y.values[[1]] - auc0@y.values[[1]]
}

library(boot)

print("Bootstrap sampling...")
bootres <- boot(dat, optim, R=nboot)
print("Done.")
bias_mean <- mean(bootres$t)
bias_std <- sd(bootres$t)
auc_corr <- auc_app - bias_mean

print("Model **************")
print(fit)

print("Performance **************")

sprintf("Apparent AUC : %0.2f", auc_app)
sprintf("Mean AUC bias: %0.2f", bias_mean)
sprintf("Std AUC bias : %0.2f", bias_std)
sprintf("Corrected AUC: %0.2f", auc_corr)
