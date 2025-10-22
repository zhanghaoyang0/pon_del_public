# =================================================
# prepare
# =================================================
import sys
sys.path.append('./code')
from train_00_function import *
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function: check if variable is binary
def is_binary(series):
    return series.dropna().nunique() == 2 and set(series.dropna().unique()).issubset({0, 1})

# Get feature importance scores
final_model = load_model("PON_Del", "all")
feature_importance = final_model.feature_importance()
feature_names = final_model.feature_name()

# Feature name mapping to formal names
feature_name_map = {
    'Haploinsufficient': 'Haploinsufficient',
    'Conservation': 'Conservation',
    'vlow_conf': 'Low-confidence SS',
    'Accessibility': 'Accessibility',
    'p_repeat': 'Protein repeat region',
    'protein_len': 'Protein length',
    'aaindex_RACS820103': 'AAindex RACS820103',
    'Closeness': 'log10(Closeness)',
    'aaindex_NAKH920103': 'AAindex NAKH920103',
    'aaindex_NAKH900104': 'AAindex NAKH900104',
    'p_palindrome': 'Palindrome region',
    'aaindex_BIGC670101': 'AAindex BIGC670101',
    'T/C': 'Turn/Coil SS',
    'aaindex_SUEM840102': 'AAindex SUEM840102',
    'aaindex_LEVM760103': 'AAindex LEVM760103',
    'aaindex_CHOP780204': 'AAindex CHOP780204',
    'aaindex_ARGP820102': 'AAindex ARGP820102',
    'aaindex_GEOR030105': 'AAindex GEOR030105',
    'Hub_Score': 'log10(Hub Score)',
    'p_domain': 'Domain localisation'
}


# =================================================
# df for plot
# =================================================
# data for shap plot

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

X_train_transformed, X_test_transformed = process_X(X_train, X_test, model_name='LightGBM')

# Use only the features that were used in the final model
model_features = feature_names
X_train_transformed = X_train_transformed[model_features]

# data for distribution plot
# Create plot data
plot_df = X_train_transformed.copy()
plot_df.columns = [feature_name_map.get(col, col) for col in plot_df.columns]  # Rename features

# Apply transformations
# plot_df['Betweenness (log10)'] = np.log10(plot_df['Betweenness (log10)'])  # Log transform
# plot_df['Deletion genomic start (Mb)'] = plot_df['Deletion genomic start (Mb)'] / 1e6  # Convert to Mb

# Crop Hub Score and Closeness at 95th percentile
plot_df['log10(Hub Score)'] = np.log10(plot_df['log10(Hub Score)'])
plot_df['log10(Closeness)'] = np.log10(plot_df['log10(Closeness)'])

plot_df['class'] = y_train.values
plot_df['class'] = plot_df['class'].map({0: 'B', 1: 'P'})  # Set class labels

# =================================================
# SHAP Analysis
# =================================================
# Calculate SHAP values
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_transformed)[1]

# Rename features in the DataFrame for plotting
X_train_transformed_plot = X_train_transformed.copy()
X_train_transformed_plot.columns = [feature_name_map.get(col, col) for col in X_train_transformed_plot.columns]

# Calculate consistent figure dimensions
n_features = len(X_train_transformed_plot.columns)

# Create layered violin plot with coolwarm color scheme
plt.figure(figsize=(2, 14))
shap.summary_plot(
    shap_values, 
    X_train_transformed_plot, 
    plot_type="layered_violin",
    color="coolwarm",
    show=False,
    max_display=30
)
# plt.xlim(-0.3, 0.2) 
plt.xlabel("SHAP value")
plt.savefig('./plot/feat_shap.png', bbox_inches='tight', dpi=350)
plt.close()

# Get feature order from SHAP plot
shap_order = X_train_transformed_plot.columns.tolist()

# Reorder features to show binary ones first, then specific continuous features, then remaining ones
binary_features = [f for f in shap_order if is_binary(plot_df[f])]
continuous_features = [f for f in shap_order if not is_binary(plot_df[f])]

# Define the priority order for continuous features
priority_continuous = ['Conservation', 'Accessibility', 'Protein length', 
                      'log10(Closeness)', 'log10(Hub Score)']

# Get remaining continuous features not in the priority list
remaining_continuous = [f for f in continuous_features if f not in priority_continuous]

# Final order: binary features + priority continuous + remaining continuous
ordered_features = binary_features + priority_continuous + remaining_continuous

n_features = len(ordered_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

# Use the same height ratio for distribution plots
plt.figure(figsize=(8, 15))  # Increased height from 15 to 20

# Helper function: plot bar chart for binary features
def plot_binary_stacked_bar(data, feature, ax):
    # Calculate proportions for each class
    props = data.groupby('class')[feature].mean()
    # Plot as regular bar plot
    props.plot(kind='bar', ax=ax, color='gray', alpha=0.7)
    ax.set_ylabel('', fontsize=0)
    ax.set_title(feature, fontsize=14)
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_xticklabels(['B', 'P'], rotation=0, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

# Plotting loop
for idx, feature in enumerate(ordered_features, 1):
    ax = plt.subplot(n_rows, n_cols, idx)
    if is_binary(plot_df[feature]):
        plot_binary_stacked_bar(plot_df, feature, ax)
    else:
        sns.violinplot(data=plot_df, x='class', y=feature, inner='box', palette='Set1', cut=0, ax=ax)
        ax.set_ylabel('', fontsize=0)
        ax.set_title(feature, fontsize=14)
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_xticklabels(['B', 'P'], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)  # Ensure x-axis tick labels are consistent size

# Layout and save
plt.tight_layout()
plt.savefig('./plot/feat_distributions.png', bbox_inches='tight', dpi=400)
plt.close()


