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
load("gp_data.RData")

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

y_train <- f_train + 0.1*rnorm(100)

train_df <- data.frame(x = x_train, y = y_train)

tmp_fit1 <- rpart(y~x, data = train_df)
fit1 <- prune(tmp_fit1, cp = 0.01)
yhat_train <- predict(tmp_fit1, newdata = data.frame(x = x_train))
plot(x_train, f_train, type = "l")
lines(x_train, yhat_train, col = 'red')

bart_fit1 <- BART::wbart(x.train = x_train, y.train = y_train, ntree = 1)

lines(x_train, bart_fit1$yhat.train.mean, col = 'blue')
