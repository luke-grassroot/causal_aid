import pandas as pd
import numpy as np

import statsmodels.api as sm
from dowhy import CausalModel
from IPython.display import Image, display

import util.load as load_util
import util.explore as explore_util

sector_outcome_indicators = {
  'education': [
      'SE.PRM.NENR', 
      'SE.PRM.CMPT.ZS', 
      'SE.PRM.PRSL.ZS'
  ]
}

sector_input_variables = {
  'education': [
      'SP.POP.TOTL', 
      'SP.POP.0014.TO.ZS', # young population
      'NY.GDP.PCAP.KD',
      'NY.GDP.PCAP.PP.KD', # gdp per cap PPP
      'FP.CPI.TOTL.ZG', # inflation
      'NE.TRD.GNFS.ZS', # trade share of GDP,
  #     'SE.PRM.ENRL.TC.ZS', # teacher-pupil share (? - should be in outcome)
      'SE.XPD.TOTL.GB.ZS', # govt share of education,
      'GC.NLD.TOTL.GD.ZS' # note: discontinued cash surplus/balance, and not in MDG, so use this
  ]
}

def perform_causal_estimate(df, causal_graph, column_dict, estimator_method="backdoor.linear_regression", verbose=False):
    causal_df = df[column_dict["features"]]
    
    pre_drop_N = len(causal_df)
    causal_df = causal_df.dropna()
    post_drop_N = len(causal_df)
    
    if verbose:
        print("N pre NA drop: ", pre_drop_N, " and post: ", post_drop_N)
    
    model = CausalModel(
        data=causal_df,
        graph=causal_graph.replace("\n", " "),
        treatment=column_dict["treatment"],
        outcome=column_dict["outcome"]
    )
    
    if verbose:
        model.view_model()
        display(Image(filename="causal_model.png"))
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    if verbose:
        print(identified_estimand)
    
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression",
                                     target_units="att", test_significance=True)
    if verbose:
        print(estimate)
    
    return {"model": model, "estimand": identified_estimand, "estimate": estimate}

def find_latest_govt_rating(project_df, country, year):
    country_proj = project_df[
        (project_df['country_code'] == country) 
        & (project_df['start_year'] <= year)
        & (project_df['wb_government_partner_rating'].notna())]
    if len(country_proj) == 0:
        return np.nan
    else:
        proj_year = country_proj[country_proj.start_year == country_proj.start_year.max()]
        return proj_year['wb_government_partner_rating'].sort_values(ascending=False).iloc[0]

def assemble_replication_panel(sector, reload=True, verbose=False, add_govt_rating=False, save_df=False):
  project_df = load_util.load_projects()
  ddf = pd.read_csv('../data/countrypanel.csv')

  feature_cols = [
      'country_code', 
      'donor_name', 
      'aiddata_sectorname', 
      'six_overall_rating', 
      'start_date', 
      'completion_date', 
      'project_duration',
      'sector'
  ]

  pdf = load_util.narrow_convert_project_data(project_df, feature_cols)
  earliest_year = pdf.start_year.min()
  latest_year = pdf.start_year.max()
  year_range = range(int(earliest_year), int(latest_year))

  df, tdf = explore_util.assemble_sectoral_df(
    pdf, 
    sector_outcome_indicators['education'],
    year_range,
    interpolate_limit=5,
    persisted_lag_table=f'../data/transformed_data/{sector}_df.csv' if reload else None
  )
  # print('1. Value counts: ', df.project_completed_year.value_counts())

  ccode_corr = pd.read_csv('../data/PPD_WDI_country_codes.csv')
  if not 'ppd_countrycode' in df.columns:
    df = df.merge(ccode_corr[['ppd_countrycode', 'wdi_countrycode', 'wdi_countryname']], left_on='country',
                   right_on='ppd_countrycode', how='left')
    df['country'] = df['wdi_countrycode']
    df = df[df['country'].notna()].drop(columns=['wdi_countrycode'])

  edu_proj, df = explore_util.assemble_sector_proj_df(project_df, sector.capitalize(), df)
  # print('2. Value counts: ', df.project_completed_year.value_counts())
  df = explore_util.construct_sector_lag_table(df, tdf, pdf, sector_outcome_indicators[sector], sector)

  if verbose:
    print("Check, pairs in country projects: ", len(edu_proj.groupby(['country_code', 'end_year'])), 
        " vs true in end years: ", df.project_completed_year.value_counts()[1])
    print('Total projects: ', len(pdf))
    print('Years with sufficient coverage on lag: ', len(df[(df.education_lag_4_count > 0)]))
    # df.project_completed_year.value_counts()

  df = df.merge(ddf, how='left', left_on=['country', 'year'], right_on=['countrycode', 'year'])

  if add_govt_rating and 'most_recent_govt_rating' not in df:
    df['most_recent_govt_rating'] = df.apply(lambda row: find_latest_govt_rating(row['country'], row['year']), axis=1)

  if save_df:
      df.to_csv(f'../data/transformed_data/{sector}_df.csv', index=False)

  return df, ddf

def extract_culprit_counts(df, data_cols):
    null_df = df.isna()
    null_df['number_missing'] = null_df[data_cols].sum(axis=1)
    clean_number_base = null_df['number_missing'].value_counts().to_dict()[0]
    culprit_counts = {}
    for test_col in data_cols:
        tmp_cols = [col for col in data_cols if col != test_col]
        temp_df = null_df.drop(columns=[test_col])
        temp_df['number_missing'] = temp_df[tmp_cols].sum(axis=1)
        new_clean = temp_df['number_missing'].value_counts().to_dict()[0]
        culprit_counts[test_col] = new_clean - clean_number_base
        
    return culprit_counts, null_df


def plain_vanilla_ols(df, target_col, input_cols, add_country_feffects=False, file_to_write=None, add_constant=True):
  ols_df = df[input_cols]

  if add_country_feffects:
      ols_df = pd.concat((ols_df, pd.get_dummies(df['country'], drop_first=True)), axis=1)

  data = ols_df.dropna()

  cols_to_drop = [target_col, 'country', 'year', 'ppd_countrycode', 'wdi_countryname', 'project_start_year']
  
  y = data[target_col]
  x = data.drop(columns=cols_to_drop, errors='ignore').replace({False: 0, True: 1})

  if add_constant:
    x = sm.add_constant(x)

  est = sm.OLS(y, x).fit()

  if file_to_write is not None:
    with open(file_to_write, 'w') as file:
        file.write(est.summary().as_text())
  
  return est

def partial_out_ols(df, target_col, treatment_col, data_cols, add_country_feffects=False, add_constant=True):
  target_feature_cols = data_cols.copy()

  if treatment_col in target_feature_cols:
    target_feature_cols.remove(treatment_col)

  treatment_feature_cols = data_cols.copy()
  if target_col in treatment_feature_cols:
    treatment_feature_cols.remove(target_col)

  # note: since we need indices aligned here, we do it all internally
  data = df[data_cols]
  if add_country_feffects:
    data = pd.concat((data, pd.get_dummies(data['country'], drop_first=True)), axis=1)

  data = data.dropna().replace({ False: 0, True: 1 })

  cols_to_drop = [target_col, treatment_col, 'country', 'year', 'ppd_countrycode', 'wdi_countryname', 'project_start_year']

  y_r1 = data[target_col]
  x_r1 = data.drop(columns=cols_to_drop, errors='ignore')

  est_target = sm.OLS(y_r1, x_r1 if not add_constant else sm.add_constant(x_r1)).fit()

  y_r2 = data[treatment_col]
  x_r2 = data.drop(columns=cols_to_drop, errors='ignore')
  
  est_treatment = sm.OLS(y_r2, x_r2 if not add_constant else sm.add_constant(x_r2)).fit()

  target_resid = est_target.resid
  treatment_resid = est_treatment.resid if not add_constant else sm.add_constant(est_treatment.resid)

  dml_est = sm.OLS(target_resid, treatment_resid).fit()

  return dml_est, est_target, est_treatment

def assemble_econml_tuples(df, target_col, treatment_col, feature_cols):
  data = df[feature_cols + [treatment_col] + [target_col] + ['country']]
  data = data.dropna().replace({ False: 0, True: 1 })

  Y = data[target_col]
  T = data[treatment_col]
  X = data[feature_cols]
  W = pd.get_dummies(data['country'], drop_first=True)

  return Y, T, X, W
