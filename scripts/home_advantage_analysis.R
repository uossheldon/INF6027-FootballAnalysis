# Load required library
library(ggplot2)
library(dplyr)

# Load the data
results <- read.csv("results.csv")

# Filter data for home/away and neutral venues
home_away_data <- results %>% filter(venue == "Home/Away")
neutral_data <- results %>% filter(venue == "Neutral Venue")

# Calculate win rates
win_rates <- results %>%
  group_by(venue, result) %>%
  summarise(Count = n()) %>%
  mutate(WinRate = Count / sum(Count) * 100)

# Save win rates to a CSV file
write.csv(win_rates, "win_rates.csv", row.names = FALSE)

# Calculate average goals for home and away teams
average_goals <- results %>%
  summarise(
    HomeGoalsAvg = mean(home_score, na.rm = TRUE),
    AwayGoalsAvg = mean(away_score, na.rm = TRUE)
  )

# Perform a t-test
t_test_result <- t.test(home_score ~ away_score, data = results)

# Save average goals and t-test results
write.csv(average_goals, "average_goals.csv", row.names = FALSE)

# Create the bar plot for win rates
win_plot <- ggplot(win_rates, aes(x = venue, y = WinRate, fill = result)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparative Win Rates Across Match Venues",
       x = "Match Venue",
       y = "Win Rate (%)",
       fill = "Outcome Type") +
  theme_minimal()

# Save the plot
ggsave("win_rate_plot.png", win_plot, width = 8, height = 6)
