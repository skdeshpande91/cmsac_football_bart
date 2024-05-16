Completion Probability
================
2024-05-18

## Overview

Here we will use **BART** and **flexBART** to fit completition
probability models based on a subset of the 2019 Big Data Bowl data.
Below, we load in the data, which has already been pretty heavily
pre-processed, and create a training/testing split.

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✔ ggplot2 3.5.0     ✔ purrr   0.3.4
    ## ✔ tibble  3.1.7     ✔ dplyr   1.0.9
    ## ✔ tidyr   1.2.0     ✔ stringr 1.4.0
    ## ✔ readr   2.1.2     ✔ forcats 0.5.1

    ## Warning: package 'ggplot2' was built under R version 4.2.3

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
load("bdb2019_data.RData")
n_all <- nrow(raw_data) # total number of observations

set.seed(518)
train_index <- sort(sample(1:n_all, size = floor(0.9 * n_all)))
test_index <- (1:n_all)[-train_index]

Y_train <- raw_data$Y[train_index]
Y_test <- raw_data$Y[test_index]
```

## Fitting a model with **BART**

We’re first going to fit a model using **BART**. Since we have a binary
outcome, the standard BART approach is to fit a probit regression model

![\mathbb{P}(Y = 1) = \Phi(f(x))](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmathbb%7BP%7D%28Y%20%3D%201%29%20%3D%20%5CPhi%28f%28x%29%29 "\mathbb{P}(Y = 1) = \Phi(f(x))")

where
![\Phi](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5CPhi "\Phi")
is the cumulative distribution function of a standard normal. This is
done primarily for computational ease. In **BART**, we use the `pbart()`
function.

``` r
bart_df_all <- as.data.frame(raw_data %>% select(-Y))
bart_df_train <- bart_df_all[train_index,]
bart_df_test <- bart_df_all[test_index,]
```

We have a few categorical predictors that encode the identifies of the
receiver and the defensive players closes to the receiver when the ball
was thrown and when the catch was attempted. **BART** one-hot encodes
these variables, creating a binary indicator for each player. This adds
a considerable number of covariates and so we will use Linero (2018)’s
variant of BART that allows the trees to adaptively propose splitting
variables.

``` r
bart_chain1 <- 
  BART::pbart(x.train = bart_df_train,
              y.train = Y_train,
              x.test = bart_df_test)
bart_chain2 <- 
  BART::pbart(x.train = bart_df_train,
              y.train = Y_train,
              x.test = bart_df_test)
bart_chain3 <- 
  BART::pbart(x.train = bart_df_train,
              y.train = Y_train,
              x.test = bart_df_test)
bart_chain4 <- 
  BART::pbart(x.train = bart_df_train,
              y.train = Y_train,
              x.test = bart_df_test)
```

We will compute the posterior mean of the catch probabilities
(![\Phi(f(x))](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5CPhi%28f%28x%29%29 "\Phi(f(x))"))
in both the training and testing dataset. These are saved as
`prob.train.mean` and `prob.test.mean` in the object returned by
`pbart`.

``` r
bart_phat_train <- rowMeans(cbind(bart_chain1$prob.train.mean,
                             bart_chain2$prob.train.mean,
                             bart_chain3$prob.train.mean,
                             bart_chain4$prob.train.mean))

bart_phat_test <- rowMeans(cbind(bart_chain1$prob.test.mean,
                             bart_chain2$prob.test.mean,
                             bart_chain3$prob.test.mean,
                             bart_chain4$prob.test.mean))
```

``` r
brier <- data.frame(train = c(pbart = NA, flexBART = NA),
                    test = c(pbart = NA, flexBART = NA))
misclass <- brier
logloss <- brier


brier["pbart", "train"] <- 
  mean( (Y_train - bart_phat_train)^2)
brier["pbart", "test"] <-
  mean( (Y_test - bart_phat_test)^2)

misclass["pbart","train"] <-
  mean( Y_train != (bart_phat_train >= 0.5) )
misclass["pbart","test"] <-
  mean( Y_test != (bart_phat_test >= 0.5) )

logloss["pbart", "train"] <-
  -1 * mean(Y_train * log(bart_phat_train) + 
              (1 - Y_train) * log(1 - bart_phat_train), na.rm = TRUE)

logloss["pbart", "test"] <-
  -1 * mean(Y_test * log(bart_phat_test) + 
              (1 - Y_test) * log(1 - bart_phat_test), na.rm = TRUE)
```

## Fitting a model with **flexBART**

Above, we fit two models using `BART::pbart()`, one with
`sparse = FALSE` (the default) and one with `sparse = TRUE`. One
potential issue with these models is that they one-hot encode the
categorical inputs like receiver identity or the identifity of the
closest defender to the targetted receiver. As a result, the trees in
the BART ensemble will partition the different receivers in a “remove
one at time” fashion. This means that in the leaf of any tree, we are
fitting a model using data from either (i) a single receiver or (ii) all
but a small handful of receivers. This prevents us from “borrowing
strength” across small groups of similar receivers.

The **flexBART** package overcomes this limitation by allowing the tree
to assign multiple levels of a categorical variables to the left and
right branches of a tree. Unfortunately, running **flexBART** requires a
bit more data pre-processing, which we break down into several steps.

First, we separate the continuous from the categorical variables.

``` r
vars <- colnames(raw_data)
cat_vars <- c("receiver_id", "def_play_id", "def_pass_id", "score_diff_cat")
cont_vars <- vars[!vars%in% c(cat_vars, "Y")]
p_cont <- length(cont_vars)
p_cat <- length(cat_vars)
```

### Re-scaling continuous variables & definint cutpoints

The **BART** package creates a set of “cutpoints” at which each
continuous variable can split. **flexBART**, on the other hand,
distinguishes between two types of ordered variables: ones that are
truly continuous and ones that ordinal. In the context of this
application, things like `time_to_snap` and `separation_play` are truly
continuous while variables like `yardsToGo` and `down` are discrete but
ordered. For the truly continuous variables, we want to allow the tree
to split on any value in \[-1,1\]. But it wouldn’t really be useful to
allow trees to split `yardsToGo` at 1.5 and 1.35. In **flexBART** we use
the argument `unif_cuts` to signal whether to draw the cutpoint from a
continuous interval (`TRUE`) or a pre-defined grid of points (`FALSE`).
We further supply that grid of points with the `cutpoints_list`
argument.

``` r
cont_vars1 <- cont_vars[!cont_vars %in% c("yardsToGo", "down")]
flex_data <-
  raw_data %>%
  mutate(across(all_of(cont_vars1), scales::rescale, to = c(-1,1))) %>%
  mutate(receiver_id = as.integer(receiver_id)-1,
         def_play_id = as.integer(def_play_id)-1,
         def_pass_id = as.integer(def_pass_id)-1,
         score_diff_cat = as.integer(score_diff_cat)-1)
```

``` r
unif_cuts <- rep(TRUE, times = p_cont)
names(unif_cuts) <- cont_vars
cutpoints_list <- list()
for(j in cont_vars) cutpoints_list[[j]] <- c(0)

unif_cuts["yardsToGo"] <- FALSE
cutpoints_list[["yardsToGo"]] <- 1:40
unif_cuts["down"] <- FALSE
cutpoints_list[["down"]] <- 1:4

cat_levels_list <- list()
cat_levels_list[["receiver_id"]] <- 0:(length(all_rec_ids)-1)
cat_levels_list[["def_pass_id"]] <- 0:(length(all_def_ids)-1)
cat_levels_list[["def_play_id"]] <- 0:(length(all_def_ids)-1)
cat_levels_list[["score_diff_cat"]] <- 0:4
```

``` r
X_cont_all <- matrix(nrow = nrow(flex_data), ncol = length(cont_vars),
                     dimnames = list(c(), cont_vars))
for(j in cont_vars) X_cont_all[,j] <- flex_data[[j]]

X_cat_all <- matrix(nrow = nrow(flex_data), ncol = length(cat_vars),
                    dimnames = list(c(), cat_vars))
for(j in cat_vars) X_cat_all[,j] <- flex_data[[j]]

X_cont_train <- X_cont_all[train_index,]
X_cat_train <- X_cat_all[train_index,]

X_cont_test <- X_cont_all[test_index,]
X_cat_test <- X_cat_all[test_index,]
```

``` r
flex_chain1 <-
  flexBART::probit_flexBART(Y_train = as.integer(Y_train),
                            X_cont_train = X_cont_train,
                            X_cat_train = X_cat_train,
                            X_cont_test = X_cont_test,
                            X_cat_test = X_cat_test,
                            unif_cuts = unif_cuts,
                            cutpoints_list = cutpoints_list,
                            cat_levels_list = cat_levels_list,
                            sparse = TRUE)
```

    ## n_train = 4421 n_test = 492 p_cont = 27  p_cat = 4
    ##   MCMC Iteration: 0 of 2000; Warmup
    ##   MCMC Iteration: 200 of 2000; Warmup
    ##   MCMC Iteration: 400 of 2000; Warmup
    ##   MCMC Iteration: 600 of 2000; Warmup
    ##   MCMC Iteration: 800 of 2000; Warmup
    ##   MCMC Iteration: 1000 of 2000; Sampling
    ##   MCMC Iteration: 1200 of 2000; Sampling
    ##   MCMC Iteration: 1400 of 2000; Sampling
    ##   MCMC Iteration: 1600 of 2000; Sampling
    ##   MCMC Iteration: 1800 of 2000; Sampling
    ##   MCMC Iteration: 2000 of 2000; Sampling

``` r
flex_chain2 <-
  flexBART::probit_flexBART(Y_train = as.integer(Y_train),
                            X_cont_train = X_cont_train,
                            X_cat_train = X_cat_train,
                            X_cont_test = X_cont_test,
                            X_cat_test = X_cat_test,
                            unif_cuts = unif_cuts,
                            cutpoints_list = cutpoints_list,
                            cat_levels_list = cat_levels_list,
                            sparse = TRUE)
```

    ## n_train = 4421 n_test = 492 p_cont = 27  p_cat = 4
    ##   MCMC Iteration: 0 of 2000; Warmup
    ##   MCMC Iteration: 200 of 2000; Warmup
    ##   MCMC Iteration: 400 of 2000; Warmup
    ##   MCMC Iteration: 600 of 2000; Warmup
    ##   MCMC Iteration: 800 of 2000; Warmup
    ##   MCMC Iteration: 1000 of 2000; Sampling
    ##   MCMC Iteration: 1200 of 2000; Sampling
    ##   MCMC Iteration: 1400 of 2000; Sampling
    ##   MCMC Iteration: 1600 of 2000; Sampling
    ##   MCMC Iteration: 1800 of 2000; Sampling
    ##   MCMC Iteration: 2000 of 2000; Sampling

``` r
flex_chain3 <-
  flexBART::probit_flexBART(Y_train = as.integer(Y_train),
                            X_cont_train = X_cont_train,
                            X_cat_train = X_cat_train,
                            X_cont_test = X_cont_test,
                            X_cat_test = X_cat_test,
                            unif_cuts = unif_cuts,
                            cutpoints_list = cutpoints_list,
                            cat_levels_list = cat_levels_list,
                            sparse = TRUE)
```

    ## n_train = 4421 n_test = 492 p_cont = 27  p_cat = 4
    ##   MCMC Iteration: 0 of 2000; Warmup
    ##   MCMC Iteration: 200 of 2000; Warmup
    ##   MCMC Iteration: 400 of 2000; Warmup
    ##   MCMC Iteration: 600 of 2000; Warmup
    ##   MCMC Iteration: 800 of 2000; Warmup
    ##   MCMC Iteration: 1000 of 2000; Sampling
    ##   MCMC Iteration: 1200 of 2000; Sampling
    ##   MCMC Iteration: 1400 of 2000; Sampling
    ##   MCMC Iteration: 1600 of 2000; Sampling
    ##   MCMC Iteration: 1800 of 2000; Sampling
    ##   MCMC Iteration: 2000 of 2000; Sampling

``` r
flex_chain4 <-
  flexBART::probit_flexBART(Y_train = as.integer(Y_train),
                            X_cont_train = X_cont_train,
                            X_cat_train = X_cat_train,
                            X_cont_test = X_cont_test,
                            X_cat_test = X_cat_test,
                            unif_cuts = unif_cuts,
                            cutpoints_list = cutpoints_list,
                            cat_levels_list = cat_levels_list,
                            sparse = TRUE)
```

    ## n_train = 4421 n_test = 492 p_cont = 27  p_cat = 4
    ##   MCMC Iteration: 0 of 2000; Warmup
    ##   MCMC Iteration: 200 of 2000; Warmup
    ##   MCMC Iteration: 400 of 2000; Warmup
    ##   MCMC Iteration: 600 of 2000; Warmup
    ##   MCMC Iteration: 800 of 2000; Warmup
    ##   MCMC Iteration: 1000 of 2000; Sampling
    ##   MCMC Iteration: 1200 of 2000; Sampling
    ##   MCMC Iteration: 1400 of 2000; Sampling
    ##   MCMC Iteration: 1600 of 2000; Sampling
    ##   MCMC Iteration: 1800 of 2000; Sampling
    ##   MCMC Iteration: 2000 of 2000; Sampling

``` r
flex_phat_train <- rowMeans(cbind(flex_chain1$prob.train.mean,
                             flex_chain2$prob.train.mean,
                             flex_chain3$prob.train.mean,
                             flex_chain4$prob.train.mean))

flex_phat_test <- rowMeans(cbind(flex_chain1$prob.test.mean,
                             flex_chain2$prob.test.mean,
                             flex_chain3$prob.test.mean,
                             flex_chain4$prob.test.mean))
```

``` r
brier["flexBART", "train"] <- 
  mean( (Y_train - flex_phat_train)^2)
brier["flexBART", "test"] <-
  mean( (Y_test - flex_phat_test)^2)

misclass["flexBART","train"] <-
  mean( Y_train != (flex_phat_train >= 0.5) )
misclass["flexBART","test"] <-
  mean( Y_test != (flex_phat_test >= 0.5) )

logloss["flexBART", "train"] <-
  -1 * mean(Y_train * log(flex_phat_train) + 
              (1 - Y_train) * log(1 - flex_phat_train), na.rm = TRUE)

logloss["flexBART", "test"] <-
  -1 * mean(Y_test * log(flex_phat_test) + 
              (1 - Y_test) * log(1 - flex_phat_test), na.rm = TRUE)
```

We can now compare the results from the two fitted models:

``` r
round(brier, digits = 2)
```

    ##          train test
    ## pbart     0.11 0.10
    ## flexBART  0.09 0.09

``` r
round(logloss, digits = 2)
```

    ##          train test
    ## pbart     0.36 0.35
    ## flexBART  0.29 0.29

``` r
round(misclass, digits = 2)
```

    ##          train test
    ## pbart     0.14 0.15
    ## flexBART  0.12 0.11

Using **flexBART** led to somewhat better predictive results! We can
also look at variable selection

``` r
varcount <- rbind(flex_chain1$varcounts,
                  flex_chain2$varcounts,
                  flex_chain3$varcounts,
                  flex_chain4$varcounts)

colnames(varcount) <- c(cont_vars, cat_vars)

varprob <- colMeans(varcount >= 1)
round(varprob, digits = 3)
```

    ##       time_snap_to_pass       time_pass_to_play              time_total 
    ##                   0.155                   1.000                   0.846 
    ##         receiver_s_pass         receiver_s_play       receiver_s_change 
    ##                   0.081                   0.992                   0.997 
    ##       receiver_dir_pass       receiver_dir_play     receiver_dir_change 
    ##                   0.072                   0.040                   0.101 
    ##    ball_receiver_x_pass    ball_receiver_y_pass ball_receiver_dist_pass 
    ##                   0.208                   0.151                   0.723 
    ##    ball_receiver_x_play    ball_receiver_y_play ball_receiver_dist_play 
    ##                   0.321                   0.626                   1.000 
    ##    ball_def_pass_x_pass    ball_def_pass_y_pass ball_def_pass_dist_pass 
    ##                   0.173                   0.218                   0.869 
    ##    ball_def_play_x_play    ball_def_play_y_play ball_def_play_dist_play 
    ##                   0.804                   0.378                   0.067 
    ##         separation_pass         separation_play       separation_change 
    ##                   0.673                   0.999                   0.429 
    ##     time_remaining_half                    down               yardsToGo 
    ##                   0.047                   0.033                   0.384 
    ##             receiver_id             def_play_id             def_pass_id 
    ##                   0.027                   0.029                   0.032 
    ##          score_diff_cat 
    ##                   0.025
