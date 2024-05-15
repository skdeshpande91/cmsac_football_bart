library(tidyverse)

load("all_training.RData")

drop_cols <- c("gameId", "playId", "outcome", "time_snap", "time_pass", "time_play", 
               "tie", "score_diff", "1_score_lead", "1_score_trail", "2_score_lead", "2_score_trail", "score_diff_sgn",
               "receiver_dist_before", "receiver_dist_after", "receiver_dist_total", "receiver_dist_cum",
               "def_play_dist_total", "def_play_dist_cum")
vars <- colnames(all_training)[!colnames(all_training) %in% drop_cols]

training_df <-
  all_training %>%
  select(all_of(vars)) %>%
  mutate(score_diff_cat = factor(score_diff_cat),
         receiver_id = factor(receiver_id),
         def_play_id = factor(def_play_id),
         def_pass_id = factor(def_pass_id))

x <- as.data.frame(training_df %>% select(-Y))
y <- as.vector(training_df$Y)
chain1 <- 
  BART::pbart(x.train = x, y.train = y, 
              sparse = TRUE, 
              ndpost = 1000, nskip = 1000)
chain2 <-
  BART::pbart(x.train = x, y.train = y,
              sparse = TRUE,
              ndpost = 1000, nskip = 1000)

chain3 <- 
  BART::pbart(x.train = x, y.train = y, 
              sparse = TRUE, 
              ndpost = 1000, nskip = 1000)
chain4 <-
  BART::pbart(x.train = x, y.train = y,
              sparse = TRUE,
              ndpost = 1000, nskip = 1000)
#####################
# Combine all the varcounts



#######
# For variable selection, we need to look at the varcounts
