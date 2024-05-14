####################################################################
# Generate data using random Fourier features
# Use D = 1000 and just draw from a squared exponential kernel
# omega0, b0 and beta0 are the true data generating parameters
####################################################################

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

x_grid <- seq(-5, 5, by = 0.01)
f_grid <- f0(x_grid)

plot(x_grid, f_grid, type = "l", xlim = c(-5,5), ylim = c(-5,5))

set.seed(724)
sigma <- 1
x_all <- runif(100, -4.75, 4.75)
f_all <- f0(x_all)
y_all <- f_all + rnorm(10, mean = 0, sd = sigma)
points(x_all, y_all, pch = 16)
save(D, omega0, b0, beta0, f0, x_grid, f_grid, 
     x_all, f_all, y_all, sigma,
     file = "gp_data.RData")
