# Load required libraries
library(data.table)
library(lightgbm)
library(caret)
library(ggplot2)

# Step 1: Load Data
results <- fread("results.csv")
goalscorers <- fread("goalscorers.csv")
shootouts <- fread("shootouts.csv")

# Step 2: Process `results.csv`
# Generate match result labels: Home Win=2, Draw=1, Away Win=0
results[, score_diff := home_score - away_score]
results[, result_label := ifelse(score_diff > 0, 2, ifelse(score_diff == 0, 1, 0))]

# Calculate home win rate per team
home_win_rate <- results[, .(home_win_rate = mean(result_label == 2, na.rm = TRUE)), by = home_team]
results <- merge(results, home_win_rate, by = "home_team", all.x = TRUE)

# Calculate head-to-head statistics
head_to_head <- results[, .(
  head_to_head_wins = sum(result_label == 2, na.rm = TRUE),
  head_to_head_draws = sum(result_label == 1, na.rm = TRUE),
  head_to_head_losses = sum(result_label == 0, na.rm = TRUE),
  total_matches = .N
), by = .(home_team, away_team)]

# Merge head-to-head stats with results
results <- merge(results, head_to_head, by = c("home_team", "away_team"), all.x = TRUE)

# Calculate head-to-head win rate
results[, head_to_head_win_rate := head_to_head_wins / total_matches]
results[, head_to_head_draw_rate := head_to_head_draws / total_matches]
results[, head_to_head_loss_rate := head_to_head_losses / total_matches]
results[is.na(head_to_head_win_rate), `:=` (
  head_to_head_win_rate = 0,
  head_to_head_draw_rate = 0,
  head_to_head_loss_rate = 0
)]

# Step 3: Process `goalscorers.csv`
# Add own goal and penalty flags dynamically
goalscorers[, is_own_goal := ifelse(own_goal == TRUE, 1, 0)]
goalscorers[, is_penalty := ifelse(penalty == TRUE, 1, 0)]

# Aggregate scoring stats by team and date
team_goals <- goalscorers[, .(
  total_goals = .N,
  own_goals = sum(is_own_goal, na.rm = TRUE),
  penalties = sum(is_penalty, na.rm = TRUE)
), by = .(team, date)]

# Merge team goals with results data for both home and away teams
results <- merge(results, team_goals[, .(home_team = team, date, home_total_goals = total_goals,
                                         home_own_goals = own_goals, home_penalties = penalties)],
                 by = c("home_team", "date"), all.x = TRUE)

results <- merge(results, team_goals[, .(away_team = team, date, away_total_goals = total_goals,
                                         away_own_goals = own_goals, away_penalties = penalties)],
                 by = c("away_team", "date"), all.x = TRUE)

# Fill NA values with 0
results[is.na(home_total_goals), home_total_goals := 0]
results[is.na(away_total_goals), away_total_goals := 0]

# Step 4: Rolling Statistics for Recent Matches
# Calculate rolling statistics for recent 5 matches
results[, recent_home_wins := frollmean(result_label == 2, 5, na.rm = TRUE), by = home_team]
results[, recent_away_wins := frollmean(result_label == 2, 5, na.rm = TRUE), by = away_team]

# Step 5: Prepare Data for Model Training
features <- results[, .(
  home_win_rate, head_to_head_win_rate, head_to_head_draw_rate, head_to_head_loss_rate,
  home_total_goals, away_total_goals, recent_home_wins, recent_away_wins
)]
labels <- results$result_label

# Split into train-test sets
set.seed(42)
train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)
train_data <- features[train_idx, ]
train_labels <- labels[train_idx]
test_data <- features[-train_idx, ]
test_labels <- labels[-train_idx]

# Step 6: Train LightGBM Model
train_matrix <- lgb.Dataset(data = as.matrix(train_data), label = train_labels)
test_matrix <- lgb.Dataset(data = as.matrix(test_data), label = test_labels)

params <- list(
  objective = "multiclass",
  metric = "multi_logloss",
  num_class = 3,
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.9,
  seed = 42
)

model <- lgb.train(
  params = params,
  data = train_matrix,
  nrounds = 500,
  valids = list(test = test_matrix),
  early_stopping_rounds = 50
)

# Step 7: Model Evaluation
# preds is a probability matrix, convert it to class labels
preds <- predict(model, as.matrix(test_data))
pred_labels <- max.col(preds, ties.method = "random") - 1  # Convert to 0, 1, 2

# Align factor levels for confusionMatrix
levels <- union(unique(pred_labels), unique(test_labels))
pred_labels <- factor(pred_labels, levels = levels)
test_labels <- factor(test_labels, levels = levels)

# Generate confusion matrix
conf_mat <- confusionMatrix(pred_labels, test_labels, mode = "everything")
print(conf_mat)

# Step 8: Feature Importance
importance <- lgb.importance(model)
print(importance)

# Visualize Feature Importance
ggplot(importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Gain") +
  theme_minimal()
