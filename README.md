# International Football Match Analysis Project

## Overview

This project explores international football match trends and dynamics using a dataset of over 47,000 matches spanning from 1872 to 2017. The research focuses on uncovering insights into home advantage, temporal scoring patterns, confederation-wise performance, and match outcomes. The project incorporates advanced statistical analysis and machine learning techniques for predictive modeling.

---

## Dataset

**Dataset Name:** International Football Results  
**Source:** [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)  
**Description:** A dataset of international football matches with detailed records on scores, venues, and match types. Three key files (`results.csv`, `goalscorers.csv`, `shootouts.csv`) provide a robust basis for analysis and predictive modeling.

---

## Project Modules

### 1. Exploratory Data Analysis (EDA)

**Scripts:**

- `confederation_match_outcome_rates_table_heatmap.R`
- `first_shooter_analysis.R`
- `goal_difference_analysis.R`
- `goal_time_distribution.R`

**Key Insights:**

- Trends in match outcomes by venue and confederation.
- Timing patterns of goals and key events.

### 2. Home Advantage Analysis

**Scripts:**

- `home_advantage_analysis.R`
- `home_win_rate_analysis.R`

**Key Insights:**

- Significant advantages for home teams in scoring and winning.
- Temporal shifts in home advantage over historical eras.

### 3. Penalty and Own Goal Analysis

**Scripts:**

- `penalty_own_goal_analysis.R`

**Key Insights:**

- Analysis of penalty and own goal trends across venues.
- Their role in match outcomes.

### 4. Machine Learning for Match Outcome Prediction

**Python Implementation:** `train_model.py`  
**R Implementation:** `train_model.R`

**Process:**

- **Preprocessing:** Feature engineering and dataset transformation.
- **Modeling:** Utilized LightGBM to predict outcomes (Home Win, Draw, Away Win).
- **Evaluation:** Achieved ~82.21% accuracy (Python) and ~79.75% accuracy (R).

**Outputs:** Confusion matrices, feature importance visualizations, and evaluation metrics.

---

## How to Run the Project

1. **Clone Repository:**
   ```bash
   git clone https://github.com/your-repo/international-football-analysis.git
   cd international-football-analysis
2. **Install Dependencies:**

- **Python:**

   ```bash
   pip install -r requirements.txt
- **R:**
   ```r
   install.packages(c("data.table", "lightgbm", "caret", "ggplot2"))
3. **Prepare Data:** Place the CSV files (`results.csv`, `goalscorers.csv`, `shootouts.csv`) in the `data/` directory.

4. **Run Scripts:**
- **R:**
   ```r
   Rscript scripts/confederation_match_outcome_rates_table_heatmap.R
   Rscript scripts/first_shooter_analysis.R
   Rscript scripts/goal_difference_analysis.R
   Rscript scripts/goal_time_distribution.R
   Rscript scripts/home_advantage_analysis.R
   Rscript scripts/home_win_rate_analysis.R
   Rscript scripts/match_outcome_distribution.R
   Rscript scripts/penalty_own_goal_analysis.R
   Rscript scripts/match_outcomes_by_venue.R
- **Python Script**:
	```bash 
	python ml/train_model.py
5. **Outputs**:
-   Results, tables, and visualizations are stored in the `outputs/` directory.
---

## File Structure
    project_root/
    ├── data/
    │   ├── goalscorers.csv
    │   ├── results.csv
    │   ├── shootouts.csv
    │   ├── results_summary_classified.csv
    ├── scripts/
    │   ├── confederation_match_outcome_rates_table_heatmap.R
    │   ├── first_shooter_analysis.R
    │   ├── goal_difference_analysis.R
    │   ├── goal_time_distribution.R
    │   ├── home_advantage_analysis.R
    │   ├── home_win_rate_analysis.R
    │   ├── match_outcome_distribution.R
    │   ├── match_outcomes_by_venue.R
    │   ├── penalty_own_goal_analysis.R
    ├── ml/
    │   ├── train_model.py
    │   ├── train_model.R
    ├── outputs/
    │   ├── visuals/
    │   │   ├── Match Outcome Distribution.png
    │   │   ├── Summary of Penalty and Own Goals.png
    │   │   ├── more_visuals...
    │   ├── tables/
    │       ├── match_outcomes_by_venue.csv
    │       ├── summary_penalty_own_goals.csv
    │       ├── more_tables...
    └── README.md

## Results

-   **Visualizations:** Heatmaps, bar charts, and line graphs to illustrate trends and anomalies.
-   **Machine Learning**:
    -   **Accuracy:** Python (82.21%), R (79.75%).
    -   Top features: `head_to_head_win_rate`, `recent_home_wins`, `home_away_goals_diff`.

## Future Work

1.  Include post-2017 match data for updated insights.
2.  Expand analysis to domestic leagues for broader applicability.
3.  Explore ensemble machine learning models for improved predictions.
4.  Investigate qualitative factors such as player performance and coaching impact.
## Acknowledgments
Thanks to the contributors of the dataset and the academic community for their support and resources.
