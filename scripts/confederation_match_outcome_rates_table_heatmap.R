# Load necessary libraries
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization
library(tidyr)      # For reshaping data

# Read results data
results <- read.csv("results.csv")

# Mapping of Confederations to countries
confederation_mapping <- list(
  UEFA = c("France", "Germany", "Spain", "Italy", "United Kingdom"),
  CAF = c("Nigeria", "Egypt", "South Africa", "Algeria", "Morocco"),
  CONMEBOL = c("Brazil", "Argentina", "Chile", "Colombia", "Uruguay"),
  AFC = c("Japan", "South Korea", "Iran", "Australia", "Saudi Arabia"),
  OFC = c("New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands"),
  CONCACAF = c("USA", "Mexico", "Canada", "Costa Rica", "Honduras")
)

# Add Confederation column based on home team
results$Confederation <- NA
for (conf in names(confederation_mapping)) {
  results$Confederation[results$home_team %in% confederation_mapping[[conf]]] <- conf
}

# Calculate win rates per confederation
confederation_stats <- results %>%
  filter(!is.na(Confederation)) %>%
  group_by(Confederation) %>%
  summarise(
    TotalMatches = n(),
    HomeWinRate = mean(home_score > away_score) * 100,
    AwayWinRate = mean(home_score < away_score) * 100,
    DrawRate = mean(home_score == away_score) * 100
  )

# Reshape data for heatmap
confederation_melt <- confederation_stats %>%
  pivot_longer(cols = c(HomeWinRate, AwayWinRate, DrawRate), 
               names_to = "OutcomeType", values_to = "Rate")

# Plot heatmap
ggplot(confederation_melt, aes(x = OutcomeType, y = Confederation, fill = Rate)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confederation-Wise Match Outcome Rates", 
    x = "Outcome Type", 
    y = "Confederation", 
    fill = "Rate (%)"
  ) +
  theme_minimal()
