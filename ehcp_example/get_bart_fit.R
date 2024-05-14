library(BART)
library(flexBART)
library(dbarts)
library(coda)
load("all_training.RData")

drop_cols <- c("gameId", "playId", "Y", "outcome", "time_snap", "time_pass", "time_play", 
               "receiver_id", "def_pass_id", "def_play_id", "tie", "score_diff_cat", "score_diff")

y <- all_training[,"Y"]
x <- all_training[,!colnames(all_training) %in% drop_cols]

pbart_fit1 <- 
  BART::pbart(x.train = x, y.train = y, 
              sparse = TRUE, 
              ndpost = 1000, nskip = 1000)
pbart_fit2 <-
  BART::pbart(x.train = x, y.train = y,
              sparse = TRUE,
              ndpost = 1000, nskip = 1000)
