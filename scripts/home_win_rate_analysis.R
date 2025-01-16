# Load required libraries
library(ggplot2)      # For data visualization
library(ggrepel)      # For creating non-overlapping labels

# Step 1: Load the datasets
# Replace 'results.csv' with the actual path to your file
results <- read.csv("results.csv")

# Step 2: Inspect the structure of the dataset
# Ensure it contains columns like "year", "home_score", "away_score", etc.
str(results)

# Step 3: Prepare the dataset for analysis
# Add a "HomeWin" column: 1 if the home team wins, otherwise 0
results$HomeWin <- ifelse(results$home_score > results$away_score, 1, 0)

# Extract the "year" from the date column (assuming a "date" column exists)
results$year <- as.numeric(substr(results$date, 1, 4))  # Extract the year from the date

# Calculate the home win rate per year
home_win_rate <- aggregate(HomeWin ~ year, data = results, FUN = mean)
home_win_rate$HomeWin <- home_win_rate$HomeWin * 100  # Convert to percentage

# Step 4: Annotate historical periods
annotations <- data.frame(
  year = c(1872, 1901, 1914, 1939, 1951, 1995, 2001), # Key historical periods
  HomeWin = c(55, 60, 70, 75, 50, 52, 48),           # Approximate y-axis positions
  label = c(
    "Early Rules\n(1872-1900)", 
    "Modern Rules\n(1901-1913)",
    "WWI\n(1914-1918)", 
    "WWII\n(1939-1945)", 
    "Broadcast Era\n(1951-1994)",
    "Bosman Ruling\n(1995)", 
    "Global Era\n(2001-2017)"
  )
)

# Step 5: Plot the data with ggplot2
ggplot(home_win_rate, aes(x = year, y = HomeWin)) +
  geom_point(color = "#FA7F6F", size = 2) + # Scatter points for home win rates
  geom_smooth(
    method = "lm", 
    color = "#82B0D2", size = 1, se = TRUE # Add trend line with confidence interval
  ) +
  geom_vline(
    xintercept = c(1901, 1914, 1939, 1951, 1995, 2001), 
    color = "#999999", linetype = "dashed", size = 0.8 # Vertical dashed lines for eras
  ) +
  geom_label_repel(
    data = annotations,
    aes(x = year, y = HomeWin, label = label),
    color = "#6C3483",       # Text color for annotations
    fill = "#E7DAD2",        # Background color for annotations
    size = 4, 
    fontface = "bold",
    nudge_y = 5,   # Adjust vertical position of labels
    nudge_x = 3    # Adjust horizontal position of labels
  ) +
  labs(
    x = "Year",
    y = "Home Win Rate (%)",
    title = "Trends in Home Win Rates (1872-2017)"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 12, color = "black"),  # Customize text size and color
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold") # Center title
  )
