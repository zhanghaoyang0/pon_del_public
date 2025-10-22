# =================================================
# prepare
# =================================================
# load functions and basic modules
import sys
import itertools
import optuna
sys.path.append('./code')
from train_00_function import *
os.chdir('.')

# =================================================
# Load data
# =================================================
# Setting
# Load data
df = pd.read_csv('./data/data_filteredFeat.csv')

# table 1
cal_freq(df, ['class', 'set'])
df[df['set'] == 'train']['NP_id'].nunique()
df[df['set'] == 'test']['NP_id'].nunique()

# seq NA should not be treated as NAN
cols_to_fix = ['del_seq', 'up5seq', 'down5seq']
df[cols_to_fix] = df[cols_to_fix].fillna('NA')

# Split data
df_train = df[df['set'] == 'train']
df_test = df[df['set'] == 'test']
y_train = df_train['class'] 
y_test = df_test['class']
X_train = df_train.drop(columns=['class', 'set', 'NP_id'])  # Features (drop the target column)
X_test = df_test[X_train.columns]  # Features (drop the target column)

# Load feature importance dictionary
with open('./data/dict_lgbmFeatImp.pkl', 'rb') as f:
    dict_lgbmFeatImp = pickle.load(f)

# Get top features from dict_lgbmFeatImp (do this once)
lgbm_feats = dict_lgbmFeatImp[20]['Feature'].tolist()

# Apply feature selection to X_train_transformed
X_train_transformed = X_train[lgbm_feats]

# =================================================
# Hyperparameter tuning with Optuna
# =================================================
# Initialize results list
results = []

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': ['binary_logloss'],
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),  # keep both for now
        # Try a more conservative learning rate range
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.005, log=True),
        # Fewer leaves to avoid overly complex trees
        'num_leaves': trial.suggest_int('num_leaves', 5, 20, step=5),
        # Encourage more randomization to prevent overfitting
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9, step=0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9, step=0.1),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7, step=2),  # more frequent bagging
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 40, step=10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 0.05, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2, step=0.1),  # avoid forcing splits to have too much gain
        # Stronger regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0, step=0.3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0, step=0.3),
        'seed': seed
    }
    # Store results for this trial
    trial_results = []
    
    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_transformed, y_train)):
        # Split data
        X_train_sub = X_train_transformed.iloc[train_idx]
        X_val_sub = X_train_transformed.iloc[val_idx]
        y_train_sub = y_train.iloc[train_idx]
        y_val_sub = y_train.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train_sub, label=y_train_sub)
        val_data = lgb.Dataset(X_val_sub, label=y_val_sub, reference=train_data)
        
        # Train model with early stopping using callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # Disable logging
        ]
        
        model = lgb.train(
            param, 
            train_data, 
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Get predictions for validation set
        y_val_pred = model.predict(X_val_sub)
        
        # Calculate metrics for this fold
        fold_metrics = calculate_metrics(y_val_sub, y_val_pred)
        trial_results.append(fold_metrics)
        
        # Store results with fold and trial number
        results.append({
            'fold': fold,
            'trial': trial.number,
            'objective': param['objective'],
            'boosting_type': param['boosting_type'],
            'learning_rate': param['learning_rate'],
            'num_leaves': param['num_leaves'],
            'feature_fraction': param['feature_fraction'],
            'bagging_fraction': param['bagging_fraction'],
            'bagging_freq': param['bagging_freq'],
            'min_child_samples': param['min_child_samples'],
            'min_child_weight': param['min_child_weight'],
            'min_split_gain': param['min_split_gain'],
            'reg_alpha': param['reg_alpha'],
            'reg_lambda': param['reg_lambda'],
            **fold_metrics
        })
    
    # Calculate mean metrics across folds
    mean_metrics = {}
    for metric in ['n_Accuracy', 'n_MCC', 'n_PPV', 'n_NPV', 'n_Sensitivity', 'n_Specificity', 'n_OPM', 'AUC']:
        mean_metrics[metric] = np.mean([r[metric] for r in trial_results])
    
    # Return the metric to optimize
    return mean_metrics['AUC']

# Create study
study = optuna.create_study(direction='maximize')

# Run optimization
n_trials = 100  # Adjust this number based on your needs
study.optimize(objective, n_trials=n_trials)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by AUC
results_df = results_df.sort_values('AUC', ascending=False)

# Save results
results_df.to_csv('./result/lgbm_hyperparaOptimized.csv', index=False)

# save study
with open('./result/lgbm_hyperparaOptimized.pkl', 'wb') as f:
    pickle.dump(study, f)

# Extract best parameters
lgbm_best_params = study.best_params
lgbm_best_params['objective'] = 'binary'
lgbm_best_params['metric'] = ['binary_logloss']
lgbm_best_params['seed'] = seed 
lgbm_best_params # manually save to train_00_function.py




# =================================================
# Optimization History + Parameter Importance
# =================================================
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import optuna

# Load Optuna study
with open('./result/lgbm_hyperparaOptimized.pkl', 'rb') as f:
    study = pickle.load(f)

# Extract data
history = study.trials_dataframe()
importance = optuna.importance.get_param_importances(study)

# Optional: rename parameters for clarity
param_names = {
    'learning_rate': 'Learning Rate',
    'boosting_type': 'Boosting Type',
    'num_leaves': 'Number of Leaves',
    'reg_alpha': 'L1 Regularization',
    'reg_lambda': 'L2 Regularization',
    'bagging_fraction': 'Bagging Fraction',
    'min_child_weight': 'Min Child Weight',
    'bagging_freq': 'Bagging Frequency',
    'feature_fraction': 'Feature Fraction',
    'min_child_samples': 'Min Child Samples',
    'min_split_gain': 'Min Split Gain'
}

# Consistent matplotlib style
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 16

# Create  layout
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 3),
    gridspec_kw={'width_ratios': [1, 1]}
)

# Plot 1: Optimization History
ax1.plot(history.index, history['value'], marker='o', markersize=5, linestyle='-', linewidth=1.5, color='black')
ax1.set_xlabel('Optimization History')
ax1.set_ylabel('AUC')
ax1.set_ylim(0.85, 0.95)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Parameter Importance (horizontal bar)
sorted_params = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
param_labels = [param_names.get(k, k) for k in sorted_params.keys()]
ax2.barh(range(len(sorted_params)), list(sorted_params.values()), color='black')
ax2.set_yticks(range(len(sorted_params)))
ax2.set_yticklabels(param_labels)
ax2.set_xlabel('Hyperparameter Importance')
ax2.invert_yaxis()
ax2.grid(True, axis='x', linestyle='--', alpha=0.6)
ax2.set_xlim(0, 0.5)

# Save standalone vertical plot
plt.tight_layout(h_pad=2.0)
plt.subplots_adjust(wspace=0.5)  # Increase horizontal gap between ax1 and ax2
plt.savefig('./plot/lgbm_optimization.png', dpi=300, bbox_inches='tight')
plt.close()
