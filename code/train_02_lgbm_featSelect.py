# =================================================
# prepare
# =================================================
import sys
sys.path.append('./code')
from train_00_function import *
os.chdir('.')

# Define metrics list
metrics = ['AUC', 'OPM', 'Accuracy', 'MCC', 'PPV', 'NPV', 'Sensitivity', 'Specificity',
    'n_OPM', 'n_Accuracy', 'n_MCC', 'n_PPV', 'n_NPV', 'n_Sensitivity', 'n_Specificity',
    'TP', 'TN', 'FP', 'FN', 'n_TP', 'n_TN', 'n_FP', 'n_FN']

# =================================================
# load data
# =================================================
# Setting
# Load data
df = pd.read_csv('./data/data_filteredFeat.csv')

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

# Define feature selection ranges
n_features_list = list(range(10, 200, 10))  # Consistent range for both RFE and performance evaluation

# =================================================
# RFE Feature selection
# =================================================
# Configure LightGBM with parameters matching train_02_train.py
model = LGBMClassifier(**lgb_params, n_jobs=-1, verbose=-1)  # Suppress verbose output

# Dictionary to store results for each nFeat
dict_lgbmFeatImp = {}

for nFeat in n_features_list:
    print(f"Processing {nFeat} features")
    rfe = RFE(estimator=model, n_features_to_select=nFeat)
    # Fit the RFE selector
    X_train_sub = X_train.drop(columns=['del_seq', 'up5seq', 'down5seq'])
    selector = rfe.fit(X_train_sub, y_train)
    # Get selected features and their importances
    idx = [i for i, rank in enumerate(selector.ranking_) if rank == 1]
    feats = X_train_sub.iloc[:, idx]
    importances = selector.estimator_.feature_importances_
    # Create feature info DataFrame
    feature_info = pd.DataFrame({
        'Feature': feats.columns,
        'Importance': importances
    })
    # Sort by importance
    feature_info = feature_info.sort_values(by=['Importance'], ascending=False)
    # Store results in dictionary
    dict_lgbmFeatImp[nFeat] = feature_info

# Save dictionary
with open('./data/dict_lgbmFeatImp.pkl', 'wb') as f:
    pickle.dump(dict_lgbmFeatImp, f)

# =================================================
# RFE nfeat performance
# =================================================
# Load feature importance dictionary
with open('./data/dict_lgbmFeatImp.pkl', 'rb') as f:
    dict_lgbmFeatImp = pickle.load(f)

# Initialize results storage
results_rfe = []

for n_features in n_features_list:
    print(f"\nProcessing with {n_features} features from RFE")
    # Initialize metrics storage for this feature set
    cv_metrics = {metric: [] for metric in metrics}
    # Process data first
    X_train_processed, X_test_processed = process_X(X_train, X_test, model_name='LightGBM')
    
    # Get features selected by RFE
    selected_features = dict_lgbmFeatImp[n_features]['Feature'].tolist()
    
    # Use only selected features
    X_train_reduced = X_train_processed[selected_features]
    X_test_reduced = X_test_processed[selected_features]
    
    # Cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_reduced, y_train)):
        X_train_sub = X_train_reduced.iloc[train_index]
        X_val_sub = X_train_reduced.iloc[val_index]
        y_train_sub = y_train.iloc[train_index]
        y_val_sub = y_train.iloc[val_index]
        
        # Train model using LGBMClassifier with verbose=-1 to suppress output
        model = LGBMClassifier(**lgb_params, verbose=-1, silent=True)
        model.fit(X_train_sub, y_train_sub)
        
        # Make probability predictions
        y_val_prob = model.predict_proba(X_val_sub)[:, 1]  # Get probability of positive class
        
        # Ensure correct types before metrics calculation
        y_val_sub = np.asarray(y_val_sub, dtype=np.int64)
        y_val_prob = np.asarray(y_val_prob, dtype=np.float64)
        
        # Calculate and store metrics
        fold_metrics = calculate_metrics(y_val_sub, y_val_prob)
       
        # Store metrics for this fold
        for metric in metrics:
            if metric in fold_metrics:
                cv_metrics[metric].append(fold_metrics[metric])
    
    # Store results with all fold metrics
    results_rfe.append({
        'n_features': n_features,
        **{f'cv_{metric}': values for metric, values in cv_metrics.items()}
    })

# Convert results to DataFrame
results_rfe_df = pd.DataFrame(results_rfe)


# Create summary DataFrame with mean ± sd
summary_df = pd.DataFrame()
for n_feat in results_rfe_df['n_features']:
    row_data = {'n_features': n_feat}
    for metric in metrics:
        values = results_rfe_df[results_rfe_df['n_features'] == n_feat][f'cv_{metric}'].iloc[0]
        mean_val = np.mean(values)
        row_data[metric] = f"{round(mean_val, 3)}"
    summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)

# Save summary to CSV
summary_df.to_csv('./result/lgbm_nfeatPerf.csv', index=False)

# =================================================
# n Feature performance plot
# =================================================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# Set global style
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 12

# Define metrics to plot
plot_metrics = ['AUC', 'n_OPM', 'n_NPV', 'n_MCC', 'n_PPV', 
                'n_Accuracy', 'n_Sensitivity', 'n_Specificity']

# Create 2x4 subplot layout
fig, axes = plt.subplots(4, 2, figsize=(10, 9))  # Increased height from 8 to 10
axes = axes.flatten()

for idx, m in enumerate(plot_metrics):
    ax = axes[idx]
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Prepare data
    metric_data = [fold_metrics for fold_metrics in results_rfe_df[f'cv_{m}']]
    positions = results_rfe_df['n_features']
    
    # Create boxplot
    ax.boxplot(
        metric_data, 
        positions=positions,
        widths=5,
        showfliers=False,
        medianprops=dict(color='red', linewidth=2),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2)
    )
    
    # Set subplot title and coordinates
    title = m.replace('n_', '') if m.startswith('n_') else m
    ax.set_ylabel(title)
    ax.set_xlabel('N top features')
    ax.set_xlim(5, 195)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(positions)
    ax.tick_params(axis='x', rotation=90)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Increase row spacing
plt.subplots_adjust(wspace=0.3)  # Increase row spacing
# Save
plt.savefig('plot/lgbm_nfeatPerf.png', format='png', bbox_inches='tight')
plt.close()

# # =================================================
# # Plot feature importance
# # =================================================
# # Load feature importance dictionary
# with open('./data/dict_lgbmFeatImp.pkl', 'rb') as f:
#     dict_lgbmFeatImp = pickle.load(f)

# # Get all features (using the last n_features value from the dictionary)
# n_features = max(dict_lgbmFeatImp.keys())
# feature_info = dict_lgbmFeatImp[n_features]

# # Create figure with larger size to accommodate all features
# plt.figure(figsize=(15, len(feature_info)*0.3))  # Adjust height based on number of features

# # Create horizontal bar plot
# bars = plt.barh(range(len(feature_info)), feature_info['Importance'], 
#                 color='skyblue', edgecolor='navy')

# # Customize the plot
# plt.yticks(range(len(feature_info)), feature_info['Feature'], fontsize=8)
# plt.xlabel('Feature Importance', fontsize=12)
# plt.title(f'Feature Importance (All {len(feature_info)} Features)', fontsize=14, pad=20)

# # Add value labels on the bars
# for i, bar in enumerate(bars):
#     width = bar.get_width()
#     plt.text(width, bar.get_y() + bar.get_height()/2, 
#              f'{width:.3f}', 
#              ha='left', va='center', fontsize=8)

# # Adjust layout
# plt.tight_layout()

# # Save plot
# plt.savefig('plot/lgbm_nfeatImp.png', format='png', bbox_inches='tight', dpi=300)
# plt.close()

