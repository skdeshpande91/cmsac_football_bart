# Load the data
library(tidyverse)
load("bdb2019_data.RData")
n_all <- nrow(raw_data) # total number of observations

set.seed(518)
train_index <- sort(sample(1:n_all, size = floor(0.9 * n_all)))
test_index <- (1:n_all)[-train_index]

Y_train <- raw_data$Y[train_index]
Y_test <- raw_data$Y[test_index]

# prepare data frame for pbart
bart_df_all <- as.data.frame(raw_data %>% select(-Y))
bart_df_train <- bart_df_all[train_index,]
bart_df_test <- bart_df_all[test_index,]


# run pbart
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

# Get the predictions from pbart
bart_phat_train <- rowMeans(cbind(bart_chain1$prob.train.mean,
                                  bart_chain2$prob.train.mean,
                                  bart_chain3$prob.train.mean,
                                  bart_chain4$prob.train.mean))

bart_phat_test <- rowMeans(cbind(bart_chain1$prob.test.mean,
                                 bart_chain2$prob.test.mean,
                                 bart_chain3$prob.test.mean,
                                 bart_chain4$prob.test.mean))


# Create some containers to hold results
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

# Prepare for flexBART
vars <- colnames(raw_data)
cat_vars <- c("receiver_id", "def_play_id", "def_pass_id", "score_diff_cat")
cont_vars <- vars[!vars%in% c(cat_vars, "Y")]
p_cont <- length(cont_vars)
p_cat <- length(cat_vars)
cont_vars1 <- cont_vars[!cont_vars %in% c("yardsToGo", "down")]
flex_data <-
  raw_data %>%
  mutate(across(all_of(cont_vars1), scales::rescale, to = c(-1,1))) %>%
  mutate(receiver_id = as.integer(receiver_id)-1,
         def_play_id = as.integer(def_play_id)-1,
         def_pass_id = as.integer(def_pass_id)-1,
         score_diff_cat = as.integer(score_diff_cat)-1)

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

# Run flexBART
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

flex_phat_train <- rowMeans(cbind(flex_chain1$prob.train.mean,
                                  flex_chain2$prob.train.mean,
                                  flex_chain3$prob.train.mean,
                                  flex_chain4$prob.train.mean))

flex_phat_test <- rowMeans(cbind(flex_chain1$prob.test.mean,
                                 flex_chain2$prob.test.mean,
                                 flex_chain3$prob.test.mean,
                                 flex_chain4$prob.test.mean))

# Assess flexBART's performance
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

round(brier, digits = 2)
round(logloss, digits = 2)
round(misclass, digits = 2)

