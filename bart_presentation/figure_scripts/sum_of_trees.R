####################################################################
# Load colors
####################################################################
my_colors <- c("#999999", "#E69F00", "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
               "#8F2727")
my_rgb <- col2rgb(my_colors, alpha = FALSE)/255

####################################################################
# Load data
####################################################################
load("sumoftrees_data.RData")

png("../figures/regression_data.png", width = 9, height = 9*9/16, units = "in", res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlim = c(-5,5), ylim = c(-5,5),
     xlab = "X", ylab = "Y")
points(x_all, y_all, pch = 16)
dev.off()

####################################################################
# To generate a sum-of-trees
####################################################################
set.seed(129)
n <- 5000
x_train <- seq(-5, 5, length = n)
f_train <- f0(x_train)

y_train <- f_train + 0.05*rnorm(100)

bart_fit1 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 1)
bart_fit5 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 5)
bart_fit10 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 10)
bart_fit50 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 50)
bart_fit100 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 100)




n_leaf1 <- length(unique(bart_fit1$yhat.train[1,]))
n_leaf5 <- length(unique(bart_fit5$yhat.train[1,]))
n_leaf10 <- length(unique(bart_fit10$yhat.train[1,]))
n_leaf50 <- length(unique(bart_fit50$yhat.train[1,]))
n_leaf100 <- length(unique(bart_fit100$yhat.train[100,]))

png("../figures/tree_appx.png", width = 6, height = 6, units = "in", res = 400)
par(mar = c(3,3,2,1), mgp = c(1.8,0.5, 0), mfrow = c(2,2))
plot(1, type = "n", xlim = c(-5,5), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", 
     main = paste0("Tree appx (", n_leaf1, " leafs)"))
lines(x_train, f_train, col = my_colors[8])
lines(x_train, bart_fit1$yhat.train[1,], col = my_colors[4])


plot(1, type = "n", xlim = c(-5,5), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", 
     main = paste0("Tree appx (", n_leaf5, " leafs)"))
lines(x_train, f_train, col = my_colors[8])
lines(x_train, bart_fit5$yhat.train[1,], col = my_colors[4])


plot(1, type = "n", xlim = c(-5,5), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", 
     main = paste0("Tree appx (", n_leaf10, " leafs)"))
lines(x_train, f_train, col = my_colors[8])
lines(x_train, bart_fit10$yhat.train[1,], col = my_colors[4])


plot(1, type = "n", xlim = c(-5,5), ylim = c(-7.5,7.5),
     xlab = "x", ylab = "y", 
     main = paste0("Tree appx (", n_leaf100, " leafs)"))
lines(x_train, f_train, col = my_colors[8])
lines(x_train, bart_fit100$yhat.train[1,], col = my_colors[4])
dev.off()