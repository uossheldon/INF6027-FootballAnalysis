# Load required libraries
library(dplyr)
library(ggplot2)

# Load the CSV data file
results <- read.csv("results.csv")

# Verify required columns are present
if (!all(c("home_team", "away_team", "home_score", "away_score") %in% colnames(results))) {
  stop("The data is missing required columns: home_team, away_team, home_score, away_score")
}

# Add a column for goal difference (Home - Away)
results <- results %>%
  mutate(goal_difference = home_score - away_score)

# Compute the distribution of goal differences
goal_diff_summary <- results %>%
  group_by(goal_difference) %>%
  summarise(frequency = n()) %>%
  mutate(percentage = (frequency / sum(frequency)) * 100) %>%
  arrange(desc(frequency))

# Print the summary table
print(goal_diff_summary)

# Create a bar chart for goal difference distribution
goal_diff_plot <- ggplot(goal_diff_summary, aes(x = goal_difference, y = frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = frequency), 
            vjust = -0.5,  # Position above the bar
            size = 3) +    # Adjust text size
  labs(title = "Home Goal Difference Distribution",
       x = "Goal Difference (Home - Away)",
       y = "Frequency") +
  theme_minimal()

# Save the plot as a PNG file
ggsave("home_goal_difference_distribution.png", plot = goal_diff_plot, width = 10, height = 6)

# Display the plot in RStudio viewer
print(goal_diff_plot)
