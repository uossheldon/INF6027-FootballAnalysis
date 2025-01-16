# Load the necessary library
library(ggplot2)

# Read the CSV file containing match results
results <- read.csv("results.csv")



#################################################################
#######  Generate Table for Match Outcomes by Venue Type  #######
#################################################################


# Filter for home/away and neutral venue
home_away <- results[results$venue == "Home/Away", ]
neutral <- results[results$venue == "Neutral Venue", ]
"C:/Users/liq24am/AppData/Local/Packages/MicrosoftWindows.Client.CBS_cw5n1h2txyewy/TempState/ScreenClip/{456768FD-6E35-4F6A-B86B-E9D36D0C4608}.png"
# Calculate summary statistics for Home/Away
home_away_summary <- data.frame(
  MatchType = "Home/Away",
  MatchOutcome = c("Home Win", "Away Win", "Draw"),
  NumberOfMatches = c(
    sum(home_away$result == "H"),
    sum(home_away$result == "A"),
    sum(home_away$result == "D")
  )
)

# Calculate percentages
home_away_summary$Percentage <- round(
  home_away_summary$NumberOfMatches / sum(home_away_summary$NumberOfMatches) * 100, 1
)

# Calculate summary statistics for Neutral Venue
neutral_summary <- data.frame(
  MatchType = "Neutral Venue",
  MatchOutcome = c("Home Win", "Away Win", "Draw"),
  NumberOfMatches = c(
    sum(neutral$result == "H"),
    sum(neutral$result == "A"),
    sum(neutral$result == "D")
  )
)

# Calculate percentages
neutral_summary$Percentage <- round(
  neutral_summary$NumberOfMatches / sum(neutral_summary$NumberOfMatches) * 100, 1
)

# Combine summaries
final_summary <- rbind(home_away_summary, neutral_summary)

# Save the summary as a CSV file
write.csv(final_summary, "match_outcome_summary.csv", row.names = FALSE)


###############################################################
#######  Generate Pie Chart for Venue Type Proportions  #######
###############################################################



# Calculate proportions of Home/Away and Neutral Venue matches
venue_summary <- data.frame(
  VenueType = c("Home/Away Matches", "Neutral Venue Matches"),
  Count = c(
    sum(results$venue == "Home/Away"),
    sum(results$venue == "Neutral Venue")
  )
)

# Calculate the percentage
venue_summary$Percentage <- round(venue_summary$Count / sum(venue_summary$Count) * 100, 2)

# Create the pie chart
pie_chart <- ggplot(venue_summary, aes(x = "", y = Percentage, fill = VenueType)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(Percentage, "%")), 
            position = position_stack(vjust = 0.5), size = 5) +
  labs(title = "Proportional Distribution of Matches by Venue Type",
       fill = "Venue Type") +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)) +
  scale_fill_manual(values = c("skyblue", "salmon"))

# Save the plot as an image file
ggsave("venue_distribution_pie_chart.png", pie_chart, width = 8, height = 6)
