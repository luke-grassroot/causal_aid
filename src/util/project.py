from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, mean_absolute_percentage_error, roc_curve
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import explore as explore_util

def sum_feature_imp(category_name, feature_imp, exp_features):
    return sum([feature_imp[col] for col in exp_features if col.startswith(category_name)])

def extract_feature_imp(est, orig_features, exp_features):
    feature_imp = { col: est.feature_importances_[i] for i, col in enumerate(exp_features) }
    summed_feature_imp = { col: sum_feature_imp(col, feature_imp, exp_features) for col in orig_features }
    return feature_imp, summed_feature_imp

def fit_score_model(X, y, est, classification=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y if classification else None)
    
    est.fit(X_train, y_train)
    scores = { 'default_score': est.score(X_test, y_test) }
    if classification:
        true_pred = est.predict_proba(X_test)[:, 1]
        scores['fscore_etc'] = precision_recall_fscore_support(y_test, est.predict(X_test), average="binary")
        scores['roc_auc'] = roc_auc_score(y_test, true_pred)
        scores['roc_curve'] = roc_curve(y_test, true_pred)
    else:
        scores['mape'] = mean_absolute_percentage_error(y_test, est.predict(X_test))
    test_data = { "X_test": X_test, "y_test": y_test }
    return est, scores

def end_to_end_project_eval(all_data, sector_key_word, target_col, variables_to_lag, observed_X_cols, loan_feature_cols, 
                            regressor=RandomForestRegressor, classifier=RandomForestClassifier, inverted_outcome=False):
    sector_data = all_data.copy()
    
    for var in variables_to_lag:
        sector_data = explore_util.lag_variable_simple(sector_data, var, variables_to_lag[var])
    
    sector_data['is_sector_project'] = sector_data.all_sectors_theme_words.str.contains(sector_key_word)
    sector_data = sector_data[sector_data.is_sector_project]
    print("Sector projects data: ", len(sector_data), " versus all projects: ", len(all_data))
    
    sdata = sector_data[['id'] + observed_X_cols + [target_col]]
    sdata = sdata.dropna()
    print("Clean observations: ", len(sdata))

    # print("Pre scaling: ", sdata[observed_X_cols[:2]].describe())
    observation_scaler = StandardScaler()
    sdata[observed_X_cols] = observation_scaler.fit_transform(sdata[observed_X_cols])
    # print("Shape of endog: ", sdata[target_col].shape, " and exog: ", sm.add_constant(sdata[observed_X_cols]).shape)
    res_est = sm.OLS(endog=sdata[target_col], exog=sm.add_constant(sdata[observed_X_cols])).fit()
    print("Naive R squared of partialling out phase: ", res_est.rsquared, " and f_p: ", res_est.f_pvalue)
    # print("Post scaling: ", sdata[observed_X_cols[:2]].describe())
    
    target_resid = f"residual_target"
    sdata[target_resid] = res_est.resid
    
    forest_data = sdata[['id', target_resid]].merge(all_data[['id'] + loan_feature_cols], how='inner')
#     print(forest_data.isna().sum())
    pre_scale_target_desc = forest_data[target_resid].describe()
    print("Descriptive stats for target: ", pre_scale_target_desc)
    
    numeric_cols = forest_data.select_dtypes(include=np.number).columns.tolist()
    treatment_scaler = StandardScaler()
    forest_data[numeric_cols] = treatment_scaler.fit_transform(forest_data[numeric_cols])

    categorical_cols = [col for col in loan_feature_cols if col not in numeric_cols]
    forest_data = pd.get_dummies(forest_data, columns=categorical_cols)

    forest_data = forest_data.dropna()
    print("Clean within project characteristics: ", len(forest_data))
    
    pos_std_dev_threshold = 0.1
    forest_data[f'{target_resid}_above_threshold'] = (
        forest_data[target_resid] > pos_std_dev_threshold if not inverted_outcome else 
            forest_data[target_resid] < pos_std_dev_threshold
    )
    print("Projects with residual above mean: ", len(forest_data[forest_data[target_resid] > 0]))
    print("Projects with positive residual above threshold: ", len(forest_data[forest_data[target_resid] > pos_std_dev_threshold]))
    
    nreg = regressor()
    nest = classifier()
    
    X = forest_data.drop(columns=['id', target_resid, f'{target_resid}_above_threshold'])
    
    y_reg = forest_data[target_resid]
    y_class = forest_data[f'{target_resid}_above_threshold']
    
    reg_fit, reg_scores = fit_score_model(X, y_reg, nreg)
    bin_est, bin_scores = fit_score_model(X, y_class, nest, classification=True)
    
    all_col_imp, summed_imp = extract_feature_imp(bin_est, loan_feature_cols, X.columns)
    summed_imp = { feature: score for feature, score in sorted(summed_imp.items(), key=lambda item: item[1], reverse=True)}
    
    return {
        "partial_out_model": res_est,
        "residual_regressor": reg_fit,
        "residual_classifier": bin_est,
        "regression_scores": reg_scores,
        "classifier_scores": bin_scores,
        "pre_scale_target_stats": pre_scale_target_desc,
        "summed_importances": summed_imp,
        "all_importances": all_col_imp,
        "residual_df": forest_data
    }
