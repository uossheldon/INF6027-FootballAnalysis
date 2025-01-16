# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score

# # Load dataset
# results_path = 'results.csv'  # Replace with your file path
# results = pd.read_csv(results_path)

# # Data preparation
# results['date'] = pd.to_datetime(results['date'])
# results['year'] = results['date'].dt.year
# results['score_diff'] = results['home_score'] - results['away_score']
# # Re-encode target variable
# results['match_result'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# # Calculate yearly home win rate without triggering a warning
# home_win_rate = results.groupby('year', as_index=False).apply(
#     lambda group: pd.DataFrame({
#         'year': [group.name],
#         'home_win_rate': [(group['score_diff'] > 0).mean()]
#     })
# ).reset_index(drop=True)

# # Merge home win rate into the main dataset
# results = results.merge(home_win_rate, on='year', how='left')

# # Encode venue type
# results['neutral'] = results['neutral'].astype(int)

# # Features and target
# X = results[['home_win_rate', 'year', 'neutral']].fillna(0)
# y = results['match_result']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # XGBoost model
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# # Hyperparameter tuning with GridSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Best model from grid search
# best_model = grid_search.best_estimator_

# # Predictions and evaluation
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win'])
# best_params = grid_search.best_params_

# # Save results to text files
# with open("classification_report.txt", "w") as f:
#     f.write("Classification Report:\n")
#     f.write(report)

# with open("accuracy.txt", "w") as f:
#     f.write(f"Model Accuracy: {accuracy:.2f}\n")

# with open("best_params.txt", "w") as f:
#     f.write("Best Hyperparameters:\n")
#     for param, value in best_params.items():
#         f.write(f"{param}: {value}\n")


# import pandas as pd
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import classification_report, accuracy_score

# # Load dataset
# results = pd.read_csv('results.csv')

# # Feature engineering on results.csv
# results['date'] = pd.to_datetime(results['date'])
# results['year'] = results['date'].dt.year
# results['score_diff'] = results['home_score'] - results['away_score']
# results['match_result'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# # Add match frequency by team and year
# team_matches = results.groupby(['year', 'home_team']).size().reset_index(name='home_matches')
# results = results.merge(team_matches, left_on=['year', 'home_team'], right_on=['year', 'home_team'], how='left')

# # Calculate yearly home win rate
# home_win_rate = results.groupby('year', as_index=False).apply(
#     lambda group: pd.DataFrame({
#         'year': [group.name],
#         'home_win_rate': [(group['score_diff'] > 0).mean()]
#     })
# ).reset_index(drop=True)
# results = results.merge(home_win_rate, on='year', how='left')

# # Prepare features and target
# X = results[['home_win_rate', 'year', 'score_diff', 'home_matches']].fillna(0)
# y = results['match_result']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # LightGBM model
# lgbm_model = LGBMClassifier(class_weight='balanced', random_state=42)

# # Parameter grid for RandomizedSearchCV
# param_grid = {
#     'num_leaves': [31, 50],
#     'max_depth': [10, 20],
#     'learning_rate': [0.1],
#     'n_estimators': [50, 100],
#     'min_data_in_leaf': [20, 50]
# }

# # Randomized search
# random_search = RandomizedSearchCV(
#     estimator=lgbm_model,
#     param_distributions=param_grid,
#     n_iter=10,  # Search 10 random parameter combinations
#     scoring='accuracy',
#     cv=3,
#     n_jobs=-1,
#     random_state=42
# )
# random_search.fit(X_train, y_train)

# # Best model and evaluation
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win'])
# best_params = random_search.best_params_

# # Save results
# with open("optimized_classification_report.txt", "w") as f:
#     f.write("Classification Report:\n")
#     f.write(report)

# with open("optimized_accuracy.txt", "w") as f:
#     f.write(f"Model Accuracy: {accuracy:.2f}\n")

# with open("optimized_best_params.txt", "w") as f:
#     f.write("Best Hyperparameters:\n")
#     for param, value in best_params.items():
#         f.write(f"{param}: {value}\n")



# import pandas as pd
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # Load dataset
# results = pd.read_csv('results.csv')

# # Feature engineering
# results['date'] = pd.to_datetime(results['date'])
# results['year'] = results['date'].dt.year
# results['score_diff'] = results['home_score'] - results['away_score']
# results['match_result'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# # Add match frequency by team and year
# team_matches = results.groupby(['year', 'home_team']).size().reset_index(name='home_matches')
# results = results.merge(team_matches, left_on=['year', 'home_team'], right_on=['year', 'home_team'], how='left')

# # Calculate yearly home win rate
# home_win_rate = results.groupby('year', as_index=False).agg(
#     home_win_rate=('score_diff', lambda x: (x > 0).mean())
# )
# results = results.merge(home_win_rate, on='year', how='left')

# # Features and target (remove 'score_diff' for validation)
# X = results[['home_win_rate', 'year', 'home_matches']].fillna(0)  # Exclude 'score_diff'
# y = results['match_result']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Verify no overlap between train and test sets
# overlap = set(X_train.index).intersection(set(X_test.index))
# print(f"Overlap between train and test sets: {len(overlap)} samples")

# # Train LightGBM model
# lgbm_model = LGBMClassifier(class_weight='balanced', random_state=42)
# lgbm_model.fit(X_train, y_train)

# # Predictions and evaluation
# y_pred = lgbm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win'])

# # Save results
# with open("validation_classification_report.txt", "w") as f:
#     f.write("Classification Report:\n")
#     f.write(report)

# with open("validation_accuracy.txt", "w") as f:
#     f.write(f"Model Accuracy: {accuracy:.2f}\n")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import lightgbm as lgb

# # Step 1: Load Data
# results = pd.read_csv('results.csv')
# goalscorers = pd.read_csv('goalscorers.csv')
# shootouts = pd.read_csv('shootouts.csv')

# # Step 2: Process `results.csv`
# # 生成比赛结果标签：主场胜=2，平局=1，客场胜=0
# results['score_diff'] = results['home_score'] - results['away_score']
# results['result_label'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# # 提取特征：主场得分、客场得分、比赛是否中立场地
# results_features = results[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral', 'result_label']]

# # Step 3: Process `goalscorers.csv`
# # 统计每支球队的进球数据
# goalscorers['is_own_goal'] = goalscorers['own_goal'].apply(lambda x: 1 if x else 0)
# goalscorers['is_penalty'] = goalscorers['penalty'].apply(lambda x: 1 if x else 0)

# # 每支球队的进球统计
# team_goals = goalscorers.groupby(['team', 'date']).agg(
#     total_goals=('minute', 'count'),
#     own_goals=('is_own_goal', 'sum'),
#     penalties=('is_penalty', 'sum')
# ).reset_index()

# # Step 4: Process `shootouts.csv`
# # 提取点球大战的胜者和首发射手
# shootouts_features = shootouts[['date', 'home_team', 'away_team', 'winner', 'first_shooter']]

# # Step 5: Combine Features for Training
# # 示例：使用 results.csv 特征作为主输入
# X = results_features[['home_score', 'away_score', 'neutral']]
# y = results_features['result_label']

# # Step 6: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 7: Train LightGBM Model
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',  # 多分类问题
#     'metric': 'multi_logloss',
#     'num_class': 3,  # 类别数量：胜、平、负
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'feature_fraction': 0.9,
#     'seed': 42
# }

# # Train model
# lgbm_model = lgb.train(
#     params,
#     train_data,
#     num_boost_round=500,
#     valid_sets=[train_data, test_data],  # 验证集
#     valid_names=['train', 'test'],      # 验证集的名称
#     callbacks=[lgb.early_stopping(stopping_rounds=50),  # 使用回调函数实现早停
#                lgb.log_evaluation(period=10)]           # 每10轮显示一次日志
# )

# # Step 8: Model Evaluation
# y_pred_prob = lgbm_model.predict(X_test)
# y_pred = np.argmax(y_pred_prob, axis=1)

# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# # Step 9: Feature Importance (Optional)
# importance = lgbm_model.feature_importance(importance_type='gain')
# features = X_train.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

# print("\nFeature Importance:")
# print(importance_df)



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import lightgbm as lgb

# # Step 1: Load Data
# results = pd.read_csv('results.csv')
# goalscorers = pd.read_csv('goalscorers.csv')
# shootouts = pd.read_csv('shootouts.csv')

# # Step 2: Process `results.csv`
# # Generate match result labels: Home Win=2, Draw=1, Away Win=0
# results['score_diff'] = results['home_score'] - results['away_score']
# results['result_label'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# # Calculate home win rate per team
# home_win_rate = results.groupby('home_team')['result_label'].apply(lambda x: (x == 2).mean()).reset_index()
# home_win_rate.rename(columns={'result_label': 'home_win_rate'}, inplace=True)

# # Merge home win rate back to results
# results = results.merge(home_win_rate, on='home_team', how='left')

# # Calculate head-to-head statistics
# head_to_head = results.groupby(['home_team', 'away_team']).agg(
#     head_to_head_wins=('result_label', lambda x: (x == 2).sum()),
#     head_to_head_draws=('result_label', lambda x: (x == 1).sum()),
#     head_to_head_losses=('result_label', lambda x: (x == 0).sum()),
#     total_matches=('result_label', 'count')
# ).reset_index()

# # Merge head-to-head stats with results
# results = results.merge(head_to_head, on=['home_team', 'away_team'], how='left')

# # Calculate head-to-head win rate
# results['head_to_head_win_rate'] = results['head_to_head_wins'] / results['total_matches']
# results['head_to_head_draw_rate'] = results['head_to_head_draws'] / results['total_matches']
# results['head_to_head_loss_rate'] = results['head_to_head_losses'] / results['total_matches']

# # Fill NaN values with 0
# results.fillna({'head_to_head_win_rate': 0, 'head_to_head_draw_rate': 0, 'head_to_head_loss_rate': 0}, inplace=True)

# # Step 3: Process `goalscorers.csv`
# # Add own goal and penalty flags dynamically
# if 'own_goal' in goalscorers.columns and 'penalty' in goalscorers.columns:
#     goalscorers['is_own_goal'] = goalscorers['own_goal'].apply(lambda x: 1 if x else 0)
#     goalscorers['is_penalty'] = goalscorers['penalty'].apply(lambda x: 1 if x else 0)
# else:
#     # Create default columns if not present
#     goalscorers['is_own_goal'] = 0
#     goalscorers['is_penalty'] = 0

# # Aggregate scoring stats by team and date
# team_goals = goalscorers.groupby(['team', 'date']).agg(
#     total_goals=('minute', 'count'),
#     own_goals=('is_own_goal', 'sum'),
#     penalties=('is_penalty', 'sum')
# ).reset_index()

# # Merge team goals with results data for both home and away teams
# results = results.merge(
#     team_goals.rename(columns={'team': 'home_team', 'date': 'date', 'total_goals': 'home_total_goals',
#                                'own_goals': 'home_own_goals', 'penalties': 'home_penalties'}),
#     on=['home_team', 'date'], how='left'
# )

# results = results.merge(
#     team_goals.rename(columns={'team': 'away_team', 'date': 'date', 'total_goals': 'away_total_goals',
#                                'own_goals': 'away_own_goals', 'penalties': 'away_penalties'}),
#     on=['away_team', 'date'], how='left'
# )

# # Fill NaN values with 0 for scoring stats
# results.fillna({'home_total_goals': 0, 'home_own_goals': 0, 'home_penalties': 0,
#                 'away_total_goals': 0, 'away_own_goals': 0, 'away_penalties': 0}, inplace=True)

# # Calculate additional features
# results['home_away_goals_diff'] = results['home_total_goals'] - results['away_total_goals']
# results['home_away_win_rate_diff'] = results['home_win_rate'] - results['head_to_head_loss_rate']

# # Step 4: Process `shootouts.csv`
# # Extract shootout winner and first shooter info
# shootouts_features = shootouts[['date', 'home_team', 'away_team', 'winner', 'first_shooter']]

# # Merge shootout features with results
# results = results.merge(shootouts_features, on=['date', 'home_team', 'away_team'], how='left')

# # Step 5: Generate Extended Feature Set
# # Combine features for training
# extended_features = results[['home_win_rate', 'head_to_head_win_rate', 'head_to_head_draw_rate', 'head_to_head_loss_rate',
#                               'home_away_win_rate_diff', 'home_away_goals_diff',
#                               'home_total_goals', 'away_total_goals']]

# # Labels
# y = results['result_label']

# # Step 6: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(extended_features, y, test_size=0.2, random_state=42)

# # Step 7: Train LightGBM Model
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',  # Multi-class classification
#     'metric': 'multi_logloss',
#     'num_class': 3,  # Number of classes: Win, Draw, Loss
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'feature_fraction': 0.9,
#     'seed': 42
# }

# # Train the model
# lgbm_model = lgb.train(
#     params,
#     train_data,
#     num_boost_round=500,
#     valid_sets=[train_data, test_data],
#     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=10)]
# )

# # Step 8: Model Evaluation
# y_pred_prob = lgbm_model.predict(X_test)
# y_pred = np.argmax(y_pred_prob, axis=1)

# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# # Step 9: Feature Importance
# importance = lgbm_model.feature_importance(importance_type='gain')
# features = X_train.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

# print("\nFeature Importance:")
# print(importance_df)


# 修改后

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb


import matplotlib.pyplot as plt



# Step 1: Load Data
results = pd.read_csv('results.csv')
goalscorers = pd.read_csv('goalscorers.csv')
shootouts = pd.read_csv('shootouts.csv')

# Step 2: Process `results.csv`
# Generate match result labels: Home Win=2, Draw=1, Away Win=0
results['score_diff'] = results['home_score'] - results['away_score']
results['result_label'] = results['score_diff'].apply(lambda x: 2 if x > 0 else (1 if x == 0 else 0))

# Calculate home win rate per team
home_win_rate = results.groupby('home_team')['result_label'].apply(lambda x: (x == 2).mean()).reset_index()
home_win_rate.rename(columns={'result_label': 'home_win_rate'}, inplace=True)

# Merge home win rate back to results
results = results.merge(home_win_rate, on='home_team', how='left')

# Calculate head-to-head statistics
head_to_head = results.groupby(['home_team', 'away_team']).agg(
    head_to_head_wins=('result_label', lambda x: (x == 2).sum()),
    head_to_head_draws=('result_label', lambda x: (x == 1).sum()),
    head_to_head_losses=('result_label', lambda x: (x == 0).sum()),
    total_matches=('result_label', 'count')
).reset_index()

# Merge head-to-head stats with results
results = results.merge(head_to_head, on=['home_team', 'away_team'], how='left')

# Calculate head-to-head win rate
results['head_to_head_win_rate'] = results['head_to_head_wins'] / results['total_matches']
results['head_to_head_draw_rate'] = results['head_to_head_draws'] / results['total_matches']
results['head_to_head_loss_rate'] = results['head_to_head_losses'] / results['total_matches']

# Fill NaN values with 0
results.fillna({'head_to_head_win_rate': 0, 'head_to_head_draw_rate': 0, 'head_to_head_loss_rate': 0}, inplace=True)

# Step 3: Process `goalscorers.csv`
# Add own goal and penalty flags dynamically
if 'own_goal' in goalscorers.columns and 'penalty' in goalscorers.columns:
    goalscorers['is_own_goal'] = goalscorers['own_goal'].apply(lambda x: 1 if x else 0)
    goalscorers['is_penalty'] = goalscorers['penalty'].apply(lambda x: 1 if x else 0)
else:
    goalscorers['is_own_goal'] = 0
    goalscorers['is_penalty'] = 0

# Aggregate scoring stats by team and date
team_goals = goalscorers.groupby(['team', 'date']).agg(
    total_goals=('minute', 'count'),
    own_goals=('is_own_goal', 'sum'),
    penalties=('is_penalty', 'sum')
).reset_index()

# Merge team goals with results data for both home and away teams
results = results.merge(
    team_goals.rename(columns={'team': 'home_team', 'date': 'date', 'total_goals': 'home_total_goals',
                               'own_goals': 'home_own_goals', 'penalties': 'home_penalties'}),
    on=['home_team', 'date'], how='left'
)

results = results.merge(
    team_goals.rename(columns={'team': 'away_team', 'date': 'date', 'total_goals': 'away_total_goals',
                               'own_goals': 'away_own_goals', 'penalties': 'away_penalties'}),
    on=['away_team', 'date'], how='left'
)

# Fill NaN values with 0 for scoring stats
results.fillna({'home_total_goals': 0, 'home_own_goals': 0, 'home_penalties': 0,
                'away_total_goals': 0, 'away_own_goals': 0, 'away_penalties': 0}, inplace=True)

# Calculate additional features
results['home_away_goals_diff'] = results['home_total_goals'] - results['away_total_goals']
results['home_away_win_rate_diff'] = results['home_win_rate'] - results['head_to_head_loss_rate']

# Step 4: Rolling Statistics for Recent Matches
# Calculate rolling statistics for recent 5 matches
recent_home_wins = results.groupby('home_team')['result_label'].apply(
    lambda x: x.rolling(window=5, min_periods=1).apply(lambda y: (y == 2).mean(), raw=True)
).reset_index(level=0, drop=True)

recent_away_wins = results.groupby('away_team')['result_label'].apply(
    lambda x: x.rolling(window=5, min_periods=1).apply(lambda y: (y == 2).mean(), raw=True)
).reset_index(level=0, drop=True)

recent_home_goal_diff = results.groupby('home_team')['score_diff'].apply(
    lambda x: x.rolling(window=5, min_periods=1).sum()
).reset_index(level=0, drop=True)

recent_away_goal_diff = results.groupby('away_team')['score_diff'].apply(
    lambda x: x.rolling(window=5, min_periods=1).sum()
).reset_index(level=0, drop=True)

# Assign calculated values back to the DataFrame
results['recent_home_wins'] = recent_home_wins
results['recent_away_wins'] = recent_away_wins
results['recent_home_goal_diff'] = recent_home_goal_diff
results['recent_away_goal_diff'] = recent_away_goal_diff

# Step 5: Process `shootouts.csv`
shootouts_features = shootouts[['date', 'home_team', 'away_team', 'winner', 'first_shooter']]

# Merge shootout features with results
results = results.merge(shootouts_features, on=['date', 'home_team', 'away_team'], how='left')

# Step 6: Generate Extended Feature Set
extended_features = results[['home_win_rate', 'head_to_head_win_rate', 'head_to_head_draw_rate', 'head_to_head_loss_rate',
                              'home_away_win_rate_diff', 'home_away_goals_diff',
                              'recent_home_wins', 'recent_away_wins', 'recent_home_goal_diff', 'recent_away_goal_diff']]
y = results['result_label']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(extended_features, y, test_size=0.2, random_state=42)

# Step 8: Train LightGBM Model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 3,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'is_unbalance': True,  # Automatically adjust for class imbalance
    'seed': 42
}


lgbm_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=10)]
)

# Step 9: Model Evaluation
y_pred_prob = lgbm_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

from sklearn.metrics import ConfusionMatrixDisplay

# Visualize confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Away Win', 'Draw', 'Home Win'], cmap='Blues'
)
plt.title('Confusion Matrix for LightGBM Model', fontsize=16)
plt.grid(False)
plt.show()




# Feature Importance
importance = lgbm_model.feature_importance(importance_type='gain')
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)



# Visualize feature importance
plt.figure(figsize=(10, 6))
importance_df.sort_values(by='Importance', ascending=False).plot(
    kind='bar', x='Feature', y='Importance', color='skyblue', edgecolor='black', alpha=0.8
)
plt.title('Feature Importance in LightGBM Model', fontsize=16)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance (Gain)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()