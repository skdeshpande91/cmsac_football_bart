# Load a colorblind friendly palette
my_colors <- c("#999999", "#E69F00", "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


# A very non-linear function of one dimension
set.seed(129)
D <- 5000
omega0 <- rnorm(n = D, mean = 0, sd = 1.5)
b0 <- 2 * pi * runif(n = D, min = 0, max = 1)
beta0 <- rnorm(n = D, mean = 0, sd = 2*1/sqrt(D))

f0 <- function(x){
  if(length(x) == 1){
    phi <- sqrt(2) * cos(b0 + omega0*x)
    out <- sum(beta0 * phi)
  } else{
    phi_mat <- matrix(nrow = length(x), ncol = D)
    for(d in 1:D){
      phi_mat[,d] <- sqrt(2) * cos(b0[d] + omega0[d] * x)
    }
    out <- phi_mat %*% beta0
  }
  return(out)
}

# Visualize the function
x_grid <- seq(-5, 5, by = 0.01)
f_grid <- f0(x_grid)

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlim = c(-5,5), ylim = c(-8.5, 8.5),
     xlab = expression(x), ylab = expression(y))
lines(x_grid, f_grid, col = my_colors[8], lwd = 2)

# Generate data and plot it
set.seed(518)
sigma <- 1
x_train <- data.frame(x = sort(runif(1e4, -5, 5)))
f_train <- f0(x_train$x)

x_test <- data.frame(x = sort(runif(1e3, -5, 5)))
f_test <- f0(x_test$x)

y_train <- f_train + rnorm(1e4, mean = 0, sd = sigma)

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlim = c(-5,5), ylim = c(-8.5, 8.5),
     xlab = "x", ylab = "y", main = "One-dimensional example")
lines(x_grid, f_grid, col = my_colors[8], lwd = 2)
points(x_train[,1], y_train, pch = 16, cex = 0.2, col = my_colors[1])

# Run BART
fit1 <-
  BART::wbart(x.train = x_train,
              y.train = y_train)

# Fitted vs actual
fit_range <- c(-1.01,1.01) * max(abs(c(f_train, fit1$yhat.train.mean)))
plot(1, type = "n", xlim = fit_range, ylim = fit_range, main = "Actual vs fitted",
     xlab = "Actual", ylab = "Fitted")
points(f_train, fit1$yhat.train.mean, pch = 16, cex = 0.2, col = my_colors[1])
abline(a = 0, b = 1, col = my_colors[3])

# Summarize posterior
fit1_sum <-
  data.frame(
    MEAN = apply(fit1$yhat.train, MARGIN = 2, FUN = mean),
    L95 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.025),
    L80 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.1),
    L50 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.25),
    U50 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.75),
    U80 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.9),
    U95 = apply(fit1$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.975))

# Pointwise credible intervals
y_lim <- c(-1.01, 1.01) * max(abs(fit1_sum))
plot(1, type = "n", xlim = c(-5,5), ylim = y_lim,
     xlab = "x", ylab = "y", main= "Pointwise uncertainty quantification")
polygon(x = c(x_train$x, rev(x_train$x)),
        y = c(fit1_sum$L95, rev(fit1_sum$U95)),
        border = NA,
        col = adjustcolor(my_colors[2], alpha.f = 0.2))
polygon(x = c(x_train$x, rev(x_train$x)),
        y = c(fit1_sum$L80, rev(fit1_sum$U80)),
        border = NA,
        col = adjustcolor(my_colors[2], alpha.f = 0.2))
polygon(x = c(x_train$x, rev(x_train$x)),
        y = c(fit1_sum$L50, rev(fit1_sum$U50)),
        border = NA,
        col = adjustcolor(my_colors[2], alpha.f = 0.2))
lines(x_train$x, fit1_sum$MEAN, col = my_colors[2], lwd = 2)
lines(x_train$x, f_train, col = my_colors[8])

# MSE in-sample
mean(f_train >=fit1_sum$L95 & f_train <= fit1_sum$U95)

# Make a prediction
fit1 <-
  BART::wbart(x.train = x_train,
              y.train = y_train,
              x.test = x_test)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,2))
fit_range <- 
  c(-1.01,1.01) * max(abs(c(f_train, fit1$yhat.train.mean, fit1$yhat.test.mean)))
plot(1, type = "n", xlim = fit_range, ylim = fit_range, main = "Actual vs fitted (train)",
     xlab = "Actual", ylab = "Fitted")
points(f_train, fit1$yhat.train.mean, pch = 16, cex = 0.2, col = my_colors[1])
abline(a = 0, b = 1, col = my_colors[3])

plot(1, type = "n", xlim = fit_range, ylim = fit_range, main = "Actual vs fitted (test)",
     xlab = "Actual", ylab = "Fitted")
points(f_test, fit1$yhat.test.mean, pch = 16, cex = 0.2, col = my_colors[1])
abline(a = 0, b = 1, col = my_colors[3])

# In- and out-of-sample MSE
mse <- c(train = mean( (f_train - fit1$yhat.train.mean)^2 ),
         test = mean( (f_test - fit1$yhat.test.mean)^2))
mse


# Predict again
test_predict <- predict(object = fit1, newdata = x_test)
plot(1, type = "n", xlim = fit_range, ylim = fit_range, main = "Actual vs fitted",
     xlab = "Actual", ylab = "Fitted")
points(f_test, colMeans(test_predict), pch = 16, cex = 0.4, col = my_colors[1])
abline(a = 0, b = 1, col = my_colors[3])

# Friedman function
friedman <- function(x){
  if(ncol(x) < 5) stop("x_cont needs to have at least five columns")
  
  if(!all(abs(x-0.5) <= 0.5)){
    stop("all entries in x_cont must be between 0 & 1")
  } else{
    return(10 * sin(pi*x[,1]*x[,2]) + 20 * (x[,3] - 0.5)^2 + 10*x[,4] + 5 * x[,5])
  }
}

# Generate data
n_train <- 10000
n_test <- 1000
p <- 20
set.seed(99)
X_train <- matrix(runif(n_train*p, min = 0, max = 1), nrow = n_train, ncol = p,
                  dimnames = list(c(), paste0("X", 1:p)))
X_test <- matrix(runif(n_test * p, min = 0, max = 1), nrow = n_train, ncol = p,
                 dimnames = list(c(), paste0("X", 1:p)))

mu_train <- friedman(X_train)
mu_test <- friedman(X_test)

sigma <- 1
set.seed(99)

y <- mu_train + sigma * rnorm(n_train, mean = 0, sd = 1)

# Variable selection
chain1 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test)
chain2 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test)
chain3 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test)
chain4 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test)

varcount <- rbind(chain1$varcount,
                  chain2$varcount,
                  chain3$varcount,
                  chain4$varcount)
colnames(varcount) <- colnames(X_train)

varprob <- colMeans(varcount >= 1)
round(varprob, digits = 3)

# Sparse BART
sparse_chain1 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test, sparse = TRUE)
sparse_chain2 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test, sparse = TRUE)
sparse_chain3 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test, sparse = TRUE)
sparse_chain4 <- BART::wbart(x.train = X_train, y.train = y, x.test = X_test, sparse = TRUE)

sparse_varcount <- rbind(sparse_chain1$varcount, 
                         sparse_chain2$varcount,
                         sparse_chain3$varcount,
                         sparse_chain4$varcount)
colnames(sparse_varcount) <- colnames(X_train)
sparse_varprob <- colMeans(sparse_varcount >= 1)
round(sparse_varprob, digits = 3)

# Compare MSE for sprase = FALSE and sparse  = TRUE
dense_yhat <- 
  rowMeans(cbind(chain1$yhat.test.mean, 
                 chain2$yhat.test.mean, 
                 chain3$yhat.test.mean,
                 chain4$yhat.test.mean))

sparse_yhat <- 
  rowMeans(cbind(sparse_chain1$yhat.test.mean,
                 sparse_chain2$yhat.test.mean,
                 sparse_chain3$yhat.test.mean,
                 sparse_chain4$yhat.test.mean))

mean( (mu_test - dense_yhat)^2 )
mean( (mu_test - sparse_yhat)^2 )


# flexBART
rescaled_X_train <- X_train *2 - 1
rescaled_X_test <- X_test * 2 - 1

# Containers to store the performance results

timing <- c("flexBART" = NA, "BART" = NA)

bart_time <-
  system.time(
    bart_fit <-
      BART::wbart(x.train = X_train, y.train = y, x.test = X_test,
                  ndpost = 1000, nskip = 1000))

timing["BART"] <- bart_time["elapsed"]

flex_time <-
  system.time(
    flex_fit <- 
      flexBART::flexBART(Y_train = y, 
                         X_cont_train = rescaled_X_train, 
                         X_cont_test = rescaled_X_test))

timing["flexBART"] <- flex_time["elapsed"]


print(round(timing, digits = 2))
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(bart_fit$yhat.test.mean, flex_fit$yhat.test.mean,
     pch = 16, cex = 0.5, col = my_colors[1])