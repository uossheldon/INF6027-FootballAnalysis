#################################################################
#######  Load Required Libraries and Import Data  ###############
#################################################################

# Load required libraries
library(dplyr)  # For data manipulation
library(ggplot2)  # For visualization

# File paths
results_summary_file <- "results_summary_classified.csv"
results_file <- "results.csv"
goalscorers_file <- "goalscorers.csv"
shootouts_file <- "shootouts.csv"

#################################################################
#######  Generate Table for Match Outcomes by Venue Type  #######
#################################################################

# Load the classified summary file
results_summary <- read.csv(results_summary_file, stringsAsFactors = FALSE)

# Summarize match outcomes by tournament category and venue type
match_outcomes <- results_summary %>%
  group_by(tournament_category, venue_type) %>%
  summarise(
    total_matches = n(),
    home_win_rate = mean(home_score > away_score) * 100,
    away_win_rate = mean(home_score < away_score) * 100,
    draw_rate = mean(home_score == away_score) * 100
  ) %>%
  ungroup()

# Print the summarized table
print(match_outcomes)

# Save the summarized table as a CSV file
write.csv(match_outcomes, "match_outcomes_by_venue.csv", row.names = FALSE)

#################################################################
#######  Generate Bar Plot for Match Outcome Distribution  ######
#################################################################

# Load the three source files
results <- read.csv(results_file, stringsAsFactors = FALSE)
goalscorers <- read.csv(goalscorers_file, stringsAsFactors = FALSE)
shootouts <- read.csv(shootouts_file, stringsAsFactors = FALSE)

# Combine data: focusing only on match outcomes (win, draw, lose)
outcome_data <- results %>%
  mutate(
    Outcome = case_when(
      home_score > away_score ~ "Home Wins",
      home_score < away_score ~ "Away Wins",
      TRUE ~ "Draws"
    )
  )

# Count occurrences of each outcome type
outcome_distribution <- outcome_data %>%
  count(Outcome, name = "Number_of_Matches")

# Generate bar plot
plot <- ggplot(outcome_distribution, aes(x = Outcome, y = Number_of_Matches, fill = Outcome)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  geom_text(aes(label = Number_of_Matches), vjust = -0.3, size = 5) +
  scale_fill_manual(values = c("Home Wins" = "yellow", "Away Wins" = "skyblue", "Draws" = "coral")) +
  labs(
    title = "Match Outcome Distribution",
    x = "Outcome Type",
    y = "Number of Matches"
  ) +
  theme_minimal(base_size = 14)

# Save the plot as an image file
ggsave("match_outcome_distribution.png", plot, width = 8, height = 6)

# Print the plot
print(plot)

#################################################################
#######  Summary of Output Files Generated  #####################
#################################################################

cat("1. Match outcomes table saved as: match_outcomes_by_venue.csv\n")
cat("2. Bar plot saved as: match_outcome_distribution.png\n")
