my_colors <- c("#999999", "#E69F00", "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
f_true <- function(x){
  return( (3 - 3*cos(6*pi*x) * x^2) * (x > 0.6) - (10 * sqrt(x)) * (x < 0.25) )
}

set.seed(517)
n <- 5000
x <- sort(runif(n, min = 0, max = 1))
mu <- f_true(x)
y <- mu + rnorm(n, mean = 0, sd = 3)

ix1 <- which(x < 0.25)
ix2 <- which(x >= 0.25 & x <= 0.6)
ix3 <- which(x > 0.6)


fit <- BART::wbart(x.train = x, y.train = y)

fit_sum <- 
  data.frame(
    mean = apply(fit$yhat.train, MARGIN = 2, FUN = mean),
    L95 = apply(fit$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.025),
    U95 = apply(fit$yhat.train, MARGIN = 2, FUN = quantile, probs = 0.975))

png("demo_data.png", width = 6, height = 6, units = "in",
    res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5,0))
plot(1, type = "n", xlim = c(0,1), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", main = "Data")
points(x, y, pch = 16, cex = 0.2, col = my_colors[1])
dev.off()

png("demo_function.png", width = 6, height = 6, units = "in",
    res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5,0))
plot(1, type = "n", xlim = c(0,1), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", main = "Data")
lines(x[ix1], mu[ix1], col = my_colors[8])
lines(x[ix2], mu[ix2], col = my_colors[8])
lines(x[ix3], mu[ix3], col = my_colors[8])
points(x, y, pch = 16, cex = 0.2, col = my_colors[1])
dev.off()

png("demo_fitted.png", width = 6, height = 6, units = "in",
    res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5,0))
plot(1, type = "n", xlim = c(0,1), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", main = "BART estimate")
polygon(x = c(x, rev(x)),
        y = c(fit_sum$L95, rev(fit_sum$U95)),
        col = adjustcolor(my_colors[4], alpha.f = 0.2),
        border = NA)

lines(x[ix1], fit$yhat.train.mean[ix1], col = my_colors[4])
lines(x[ix2], fit$yhat.train.mean[ix2], col = my_colors[4])
lines(x[ix3], fit$yhat.train.mean[ix3], col = my_colors[4])

lines(x[ix1], mu[ix1], col = my_colors[8])
lines(x[ix2], mu[ix2], col = my_colors[8])
lines(x[ix3], mu[ix3], col = my_colors[8])
dev.off()


png("demo_fit.png", width = 9, height = 9*9/16, units = "in",
    res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,2))
plot(1, type = "n", xlim = c(0,1), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", main = "Data")
lines(x[ix1], mu[ix1], col = my_colors[8])
lines(x[ix2], mu[ix2], col = my_colors[8])
lines(x[ix3], mu[ix3], col = my_colors[8])
points(x, y, pch = 16, cex = 0.2, col = my_colors[1])


dev.off()
