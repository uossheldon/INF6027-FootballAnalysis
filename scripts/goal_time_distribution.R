# Load necessary libraries
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization

# Read the input dataset
data <- read.csv("shootout.csv")

# Create a function to group minutes into intervals (e.g., 0-5, 5-10, etc.)

# Define time intervals based on international football standards
# 0-45 minutes: First Half
# 45-90 minutes: Second Half
# 90+ minutes: Stoppage Time
# 90-120 minutes: Extra Time (if applicable in knockout matches)

group_minute_intervals <- function(minute) {
  interval <- cut(
    minute,
    breaks = seq(0, 95, by = 5), # Intervals of 5 minutes 
    labels = paste0("(", seq(0, 90, by = 5), ",", seq(5, 95, by = 5), "]"),
    right = TRUE,
    include.lowest = TRUE
  )
  return(interval)
}

# Add an interval column for minute grouping
data$MinuteGroup <- group_minute_intervals(data$minute)

# Calculate total goals by minute group
total_goals_by_group <- data %>%
  group_by(MinuteGroup) %>%
  summarise(Goals = n())

# Calculate goals by home and away teams for each minute group
goals_by_team_and_group <- data %>%
  group_by(MinuteGroup, GoalType) %>% # Assuming "GoalType" column has "Home" or "Away"
  summarise(Goals = n()) %>%
  ungroup()

# Save the aggregated data into new CSV files
write.csv(total_goals_by_group, "total_goals_by_group.csv", row.names = FALSE)
write.csv(goals_by_team_and_group, "goals_by_team_and_group.csv", row.names = FALSE)

# Create visualizations
# 1. Total goals by minute group
ggplot(total_goals_by_group, aes(x = MinuteGroup, y = Goals)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Goal Distribution by Minute Group (5-Min Intervals)",
       x = "Minute Group",
       y = "Number of Goals") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. Goals by home and away teams for each minute group
ggplot(goals_by_team_and_group, aes(x = MinuteGroup, y = Goals, fill = GoalType)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Home vs Away Goals by Minute Group",
       x = "Minute Group",
       y = "Number of Goals",
       fill = "Goal Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save visualizations as images
ggsave("total_goals_by_group.png", width = 10, height = 6)
ggsave("goals_by_team_and_group.png", width = 10, height = 6)

# Print a success message
print("Analysis completed. CSV files and visualizations have been saved.")
