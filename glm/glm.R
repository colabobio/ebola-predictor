# Fits the data using Generalized Linear Model (GLM) 
# https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html
# http://www.statmethods.net/advstats/glm.html

args <- commandArgs(trailingOnly = TRUE)
nboot <- as.integer(args[1])

dat <- read.table("../models/test/training-data-completed.csv", sep=",", header=TRUE)
labels <- dat["OUT"]
full <- glm(OUT~., data=dat)
mod <- step(full, data=dat, direction="backward")
form = mod$formula
summary(mod)

library(ROCR)
prob <- predict(mod, dat)
pred <- prediction(prob, dat["OUT"])
auc <- performance(pred, measure = "auc")
auc_app <- auc@y.values[[1]]

if (0.97 <= auc_app) {
  print(auc_app)
  print("AUC is too high")
  quit(save="no")
}

optim <- function(dat0, idx)
{      
  dat1 <- dat0[idx,]
  
  # This runs stepwise backward variable selection again on the bootstrap sample,
  # which might lead to a different model:
  full1 <- glm(OUT~., data=dat1)
  mod1 <- step(full1, data=dat1, direction="backward")
  
  # Fit the original model using the bootstrap data
  #mod1 <- glm(form, data=dat1)
  
  prob1 <- predict(mod1, dat1)
  prob0 <- predict(mod1, dat0)
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
print(form)
summary(mod)

# Odds ratios for each covariate, with 95% confidence interval
# http://www.ats.ucla.edu/stat/r/dae/logit.htm
exp(cbind(OR = coef(mod), confint(mod)))

print("Performance **************")

sprintf("Apparent AUC : %0.3f", auc_app)
sprintf("Mean AUC bias: %0.3f", bias_mean)
sprintf("Std AUC bias : %0.3f", bias_std)
sprintf("Corrected AUC: %0.3f", auc_corr)

