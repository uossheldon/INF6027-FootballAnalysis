library(dplyr)

# Read the dataset
goals_data <- read.csv("goalscorers.csv")

# Check column names to ensure correctness
print("Column names:")
print(colnames(goals_data))

# Convert 'penalty' and 'own_goal' columns to logical (TRUE/FALSE)
goals_data$penalty <- as.logical(goals_data$penalty)
goals_data$own_goal <- as.logical(goals_data$own_goal)

# Create a new column to indicate match type (Home, Away, or Neutral)
goals_data$match_type <- ifelse(
  goals_data$home_team == goals_data$team, "Home",
  ifelse(goals_data$away_team == goals_data$team, "Away", "Neutral")
)

# Summarize the data by match type
summary_table <- goals_data %>%
  group_by(match_type) %>%
  summarise(
    Penalty_Goals = sum(penalty, na.rm = TRUE),   # Total penalty goals
    Own_Goals = sum(own_goal, na.rm = TRUE),     # Total own goals
    Total_Goals = n()                            # Total goals
  ) %>%
  mutate(
    Penalty_Rate = round((Penalty_Goals / Total_Goals) * 100, 3),  # Penalty rate (%)
    Own_Goal_Rate = round((Own_Goals / Total_Goals) * 100, 3)      # Own goal rate (%)
  )

# Print the summary table
print("Summary of Penalty and Own Goals by Match Type:")
print(summary_table)

# Save the summary table to a CSV file
write.csv(summary_table, "summary_penalty_own_goals.csv", row.names = FALSE)
print("Summary has been saved to 'summary_penalty_own_goals.csv'")
