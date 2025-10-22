# =================================================
# prepare
# =================================================
# load functions and basic modules
import sys
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

# save 
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)

models = {
    "LightGBM": LGBMClassifier(**lgb_params, random_state=seed),
    "PON_Del": LGBMClassifier(**lgbm_best_params, random_state=seed),
    "RF": RandomForestClassifier(n_estimators=100, random_state=seed), 
    "LR": LogisticRegression(random_state=seed, class_weight="balanced"),
    "SVM": SVC(kernel='linear', probability=True, random_state=seed),
    "MLP": MLP(input_dim=X_train.shape[1]),
    "CNN": CNN(input_dim=X_train.shape[1], seq_scale=0.2),
    "GRU": GRU(input_dim=X_train.shape[1], seq_scale=0.2)
}

# Load feature importance dictionary
with open('./data/dict_lgbmFeatImp.pkl', 'rb') as f:
    dict_lgbmFeatImp = pickle.load(f)

# Get top features from dict_lgbmFeatImp (do this once)
pondel_feats =  dict_lgbmFeatImp[20]['Feature'].tolist()

# =================================================
# train
# =================================================
# metrics
cv_metrics = {model: {
    'n_OPM': [], 'n_NPV': [], 'n_MCC': [], 'n_PPV': [], 'n_Accuracy': [], 'n_Sensitivity': [], 'n_Specificity': [],
    'OPM': [], 'NPV': [], 'MCC': [], 'PPV': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
    'TP': [], 'TN': [], 'FP': [], 'FN': [], 'n_TP': [], 'n_TN': [], 'n_FP': [], 'n_FN': [],
    "Prob_Predictions": [], "True_Labels": [], "AUC": [], 'Best_Epoch': []
} for model in models.keys()}

test_metrics = {model: {
    'n_OPM': [], 'n_NPV': [], 'n_MCC': [], 'n_PPV': [], 'n_Accuracy': [], 'n_Sensitivity': [], 'n_Specificity': [],
    'OPM': [], 'NPV': [], 'MCC': [], 'PPV': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
    'TP': [], 'TN': [], 'FP': [], 'FN': [], 'n_TP': [], 'n_TN': [], 'n_FP': [], 'n_FN': [],
    "Prob_Predictions": [], "True_Labels": [], "AUC": [], 'Best_Epoch': []
} for model in models.keys()}


# Training on CV data
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    print(f"\nStarting Fold {fold}")
    # Split Data
    X_train_sub, X_val_sub = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_sub, y_val_sub = y_train.iloc[train_index], y_train.iloc[val_index]
    
    for model_name, model in models.items():
        print(f"\nProcessing {model_name} - Fold {fold}")
        # Process and transform data
        X_train_transformed, X_val_transformed = process_X(X_train_sub, X_val_sub, model_name)
        
        # Get correct input dimension after processing
        input_dim = X_train_transformed.shape[1]
        print(f"Input dimension after processing: {input_dim}")
        
        # Special handling for deep learning models
        if model_name in ["MLP", "CNN", "GRU"]:
            # Prepare sequence data for CNN and GRU
            seq_data = None
            val_seq_data = None
            if model_name in ["CNN", "GRU"]:
                seq_data = {
                    'del_seq': seq_encode(X_train_sub['del_seq']),
                    'up5seq': seq_encode(X_train_sub['up5seq']),
                    'down5seq': seq_encode(X_train_sub['down5seq'])
                }
                val_seq_data = {
                    'del_seq': seq_encode(X_val_sub['del_seq']),
                    'up5seq': seq_encode(X_val_sub['up5seq']),
                    'down5seq': seq_encode(X_val_sub['down5seq'])
                }
            
            # Train model with validation data for early stopping
            dl_model, best_epoch = train_DL(
                model_name=model_name,
                X_train_transformed=X_train_transformed, 
                y_train=y_train_sub,
                input_dim=input_dim,
                seq_data=seq_data,
                fold=fold,
                is_final=False,
                val_data=(X_val_transformed, y_val_sub, val_seq_data)
            )
            
            # Make predictions
            with torch.no_grad():
                dl_model.eval()
                # Convert validation features to numpy if DataFrame
                if isinstance(X_val_transformed, pd.DataFrame):
                    X_val_transformed = X_val_transformed.values
                X_val_tensor = torch.tensor(X_val_transformed, dtype=torch.float32)
                
                if model_name == "MLP":
                    y_val_prob = dl_model(X_val_tensor).numpy()
                else:  # CNN or GRU
                    y_val_prob = dl_model(
                        X_val_tensor,
                        val_seq_data['del_seq'],
                        val_seq_data['up5seq'],
                        val_seq_data['down5seq']
                    ).numpy()
                y_val_prob = y_val_prob.flatten()
            
            # Calculate metrics and store predictions
            fold_metrics = calculate_metrics(y_val_sub, y_val_prob)
            fold_metrics["Best_Epoch"] = best_epoch if best_epoch is not None else 0
            fold_metrics["Prob_Predictions"] = y_val_prob
            fold_metrics["True_Labels"] = y_val_sub
            for metric, value in fold_metrics.items():
                cv_metrics[model_name][metric].append(value)
        else:
            # Handle non-DL models
            clf = load_model(model_name, fold)
            if clf is None:
                print(f"Training new {model_name} model for fold {fold}")
                if model_name in ["LightGBM", "PON_Del"]:
                    # Use only top features for training and validation
                    if model_name == "PON_Del":
                        X_train_transformed = X_train_transformed[pondel_feats]
                        X_val_transformed = X_val_transformed[pondel_feats]
                    
                    train_data = lgb.Dataset(X_train_transformed, label=y_train_sub)
                    val_data = lgb.Dataset(X_val_transformed, label=y_val_sub, reference=train_data)
                    # Use appropriate parameters based on model name
                    params = lgb_params if model_name == "LightGBM" else lgbm_best_params
                    # Add early stopping using callbacks
                    callbacks = [
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=0)  # Disable logging
                    ]
                    
                    clf = lgb.train(
                        params, 
                        train_data, 
                        num_boost_round=1000,
                        valid_sets=[val_data],
                        callbacks=callbacks
                    )
                else:
                    clf = model.fit(X_train_transformed, y_train_sub)
                # Save the model
                save_model(clf, model_name, fold)
                print(f"Saved {model_name} model for fold {fold}")
            else:
                if model_name in ["PON_Del"]:
                    # Apply feature selection to validation data
                    X_val_transformed = X_val_transformed[pondel_feats]
            
            # Make predictions
            if model_name in ["LightGBM", "PON_Del"]:
                y_val_prob = clf.predict(X_val_transformed)
            else:
                y_val_prob = clf.predict_proba(X_val_transformed)[:, 1]
            
            # Verify prediction shape
            assert len(y_val_prob) == len(y_val_sub), f"Prediction length mismatch: pred={len(y_val_prob)}, true={len(y_val_sub)}"
            
            # Calculate and store metrics
            fold_metrics = calculate_metrics(y_val_sub, y_val_prob)
            fold_metrics["Best_Epoch"] = 0  # Non-DL models don't use epochs
            fold_metrics["Prob_Predictions"] = y_val_prob
            fold_metrics["True_Labels"] = y_val_sub
            
            for metric, value in fold_metrics.items():
                cv_metrics[model_name][metric].append(value)

        

# Training on all train data
fold = 'all'
for model_name, model in models.items():
    print(f"Processing {model_name} - Final Training")
    X_train_transformed, X_test_transformed = process_X(X_train, X_test, model_name)
    
    # Get correct input dimension after processing
    input_dim = X_train_transformed.shape[1]
    print(f"Final training input dimension: {input_dim}")
    
    if model_name in ["MLP", "CNN", "GRU"]:
        # Calculate average best epoch from CV metrics
        cv_best_epochs = cv_metrics[model_name]["Best_Epoch"]
        if len(cv_best_epochs) > 0 and any(cv_best_epochs):
            avg_best_epoch = int(np.mean([e for e in cv_best_epochs if e > 0]))
            print(f"Using average best epoch from CV: {avg_best_epoch}")
        else:
            avg_best_epoch = 300  # Default if no CV data available
            print("No CV data available, using default epochs: 300")
            
        # Prepare sequence data for CNN and GRU
        seq_data = None
        if model_name in ["CNN", "GRU"]:
            seq_data = {
                'del_seq': seq_encode(X_train['del_seq']),
                'up5seq': seq_encode(X_train['up5seq']),
                'down5seq': seq_encode(X_train['down5seq'])
            }
            
        # Train model for final training (no early stopping)
        dl_model, _ = train_DL(
            model_name=model_name,
            X_train_transformed=X_train_transformed, 
            y_train=y_train,
            input_dim=input_dim,
            seq_data=seq_data,
            fold=fold,
            epochs=avg_best_epoch,
            is_final=True
        )
        
        # Make predictions
        with torch.no_grad():
            dl_model.eval()
            # Convert test features to numpy if DataFrame
            if isinstance(X_test_transformed, pd.DataFrame):
                X_test_transformed = X_test_transformed.values
            X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)
            
            if model_name == "MLP":
                y_test_prob = dl_model(X_test_tensor).numpy()
            else:  # CNN or GRU
                test_seq_data = {
                    'del_seq': seq_encode(X_test['del_seq']),
                    'up5seq': seq_encode(X_test['up5seq']),
                    'down5seq': seq_encode(X_test['down5seq'])
                }
                y_test_prob = dl_model(
                    X_test_tensor,
                    test_seq_data['del_seq'],
                    test_seq_data['up5seq'],
                    test_seq_data['down5seq']
                ).numpy()
            y_test_prob = y_test_prob.flatten()
            
            # Add final training epochs to metrics
            test_metrics[model_name]["Best_Epoch"] = [avg_best_epoch]
    else:
        # Handle non-DL models (existing code)
        final_model = load_model(model_name, fold)
        if final_model is None:
            print(f"Training new final {model_name} model")
            if model_name in ["LightGBM", "PON_Del"]:
                # Use only top features for training and test
                if model_name == "PON_Del":
                    X_train_transformed = X_train_transformed[pondel_feats]
                    X_test_transformed = X_test_transformed[pondel_feats]
                
                train_data_final = lgb.Dataset(X_train_transformed, label=y_train)
                # Use appropriate parameters based on model name
                params = lgb_params if model_name == "LightGBM" else lgbm_best_params
                final_model = lgb.train(params, train_data_final, num_boost_round=1000)
            else:
                final_model = model.fit(X_train_transformed, y_train)
            # Save the model
            save_model(final_model, model_name, fold)
        else:
            print(f"Loaded existing final {model_name} model")
            if model_name in ["PON_Del"]:
                X_test_transformed = X_test_transformed[pondel_feats]
        
        # Make predictions
        if model_name in ["LightGBM", "PON_Del"]:
            y_test_prob = final_model.predict(X_test_transformed)
        else:
            y_test_prob = final_model.predict_proba(X_test_transformed)[:, 1]
    
    # Verify prediction shape
    assert len(y_test_prob) == len(y_test), f"Prediction length mismatch: pred={len(y_test_prob)}, true={len(y_test)}"
    
    # Calculate and store metrics
    metrics_dict = calculate_metrics(y_test, y_test_prob)
    # Ensure all values are stored as lists
    test_metrics[model_name] = {k: [v] if not isinstance(v, list) else v for k, v in metrics_dict.items()}
    test_metrics[model_name]["Best_Epoch"] = [avg_best_epoch if model_name in ["MLP", "CNN"] else 0]
    test_metrics[model_name]["Prob_Predictions"] = [y_test_prob]
    test_metrics[model_name]["True_Labels"] = [y_test]

# Save metrics
with open('result/cv_metrics.pkl', 'wb') as f:
    pickle.dump(cv_metrics, f)

with open('result/test_metrics.pkl', 'wb') as f:
    pickle.dump(test_metrics, f)


# Summary: mean and standard deviation for each metric
cv_metrics['SVM'].keys()
summary_cv = pd.DataFrame({
    model: {
        metric: f"{round(np.mean([v for v in values if not np.isnan(v)]), 2)}"
        for metric, values in metrics.items()
        if metric not in ['Prob_Predictions', 'True_Labels', 'Best_Epoch']
    }
    for model, metrics in cv_metrics.items()
}).T

summary_test = pd.DataFrame({
    model: {
        metric: f"{round(values[0], 2)}" if isinstance(values, list) else f"{round(np.mean([v for v in values if not np.isnan(v)]), 2)}"
        for metric, values in metrics.items()
        if metric not in ['Prob_Predictions', 'True_Labels', 'Best_Epoch']
    }
    for model, metrics in test_metrics.items()
}).T

# check
summary_cv[['AUC']].sort_values(by='AUC', ascending=False)
summary_test[['AUC']].sort_values(by='AUC', ascending=False)


# Combine normalized and non-normalized metrics in summary
summary_cv.insert(0, "Type", "CV")
summary_test.insert(0, "Type", "Test")
summary = pd.concat([summary_cv, summary_test], axis=0).reset_index().rename(columns={'index': 'Model'})

metrics_to_combine = ['OPM', 'PPV', 'NPV', 'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'TP', 'TN', 'FP', 'FN']
for metric in metrics_to_combine:
    if f'n_{metric}' in summary.columns and metric in summary.columns:
        summary[f'{metric} (n)'] = summary.apply(lambda x: f"{x[metric]} ({x[f'n_{metric}']})", axis=1)
        summary = summary.drop(columns=[metric, f'n_{metric}'])

# Reorder columns to keep Type and Model first
cols = ['Type', 'Model'] + [col for col in summary.columns if col not in ['Type', 'Model']]
summary = summary[cols]

summary.to_csv('result/performance.csv', index=False)
summary

