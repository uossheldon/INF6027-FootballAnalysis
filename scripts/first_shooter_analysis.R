# Load necessary libraries
library(dplyr)
library(ggplot2)

# Read the shootout.csv file
shootout <- read.csv("shootout.csv")

# Check if all required columns are present
if (!all(c("date", "home_team", "away_team", "winner", "first_shooter") %in% colnames(shootout))) {
  stop("The file is missing necessary columns: date, home_team, away_team, winner, first_shooter")
}

# Group the data by first_shooter and winner, calculate frequency and percentage
summary_results <- shootout %>%
  group_by(first_shooter, winner) %>%
  summarise(Frequency = n(), .groups = "drop") %>%
  mutate(Percentage = round((Frequency / sum(Frequency)) * 100, 2))

# Print the summary results
print(summary_results)

# Create a bar chart to compare the relationship between first_shooter and match outcomes
ggplot(summary_results, aes(x = first_shooter, y = Frequency, fill = winner)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "First Shooter and Match Outcomes",
       x = "First Shooter",
       y = "Frequency",
       fill = "Match Winner") +
  theme_minimal() +
  geom_text(aes(label = paste0(Frequency, " (", Percentage, "%)")), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3)

# Save the bar chart as a PNG file
ggsave("first_shooter_outcome_analysis.png")

# Additional analysis: calculate overall distribution of first_shooter
first_shooter_distribution <- shootout %>%
  group_by(first_shooter) %>%
  summarise(Frequency = n(), .groups = "drop") %>%
  mutate(Percentage = round((Frequency / sum(Frequency)) * 100, 2))

# Print the overall first_shooter distribution
print(first_shooter_distribution)

# Save the summary results and distribution to CSV files
write.csv(summary_results, "summary_results.csv", row.names = FALSE)
write.csv(first_shooter_distribution, "first_shooter_distribution.csv", row.names = FALSE)
