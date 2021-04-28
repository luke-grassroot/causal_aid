import pandas as pd
import numpy as np

# some helper methods mostly to improve legibility in main notebooks
def load_wdi(data_path='../data/wdi/WDIData.csv', series_path='../data/wdi/WDISeries.csv'):
  return pd.read_csv(data_path, low_memory=False), pd.read_csv(series_path, low_memory=False)

def extract_indicators(wdi_df, indicator_names):
  return wdi_df[wdi_df['Indicator Code'].isin(indicator_names)]

def load_projects(data_path="../data/aid_projects.csv"):
  return pd.read_csv(data_path, low_memory=False)

def narrow_convert_project_data(df, relevant_cols, start_date_col='start_date', end_date_col='completion_date', drop_na_end_date=True):
  df = df[relevant_cols]
  df['start_dt'] = pd.to_datetime(df[start_date_col], format='%d%b%Y', errors='coerce')
  df['start_year'] = df['start_dt'].dt.year
  df['completion_dt'] = pd.to_datetime(df[end_date_col], format='%d%b%Y', errors='coerce')
  df['end_year'] = df['completion_dt'].dt.year
  if drop_na_end_date:
    df = df.dropna(subset=['end_year'])
  return df


# utilities for extracting prior and post on indicators
def avg_indicator_prior(indicator_series, base_year, country_code, num_years=5, min_year=1960):
  series_country = indicator_series[(indicator_series['Country Code'] == country_code)]
  years_for_growth = [str(base_year - (i + 1)) for i in range(num_years)]
  years_for_growth = [year for year in years_for_growth if int(year) > min_year]
  growth_figures = series_country.iloc[0][years_for_growth].to_list()
  growth_figures = [gf for gf in growth_figures if not np.isnan(gf)]
  return sum(growth_figures) / len(growth_figures) if len(growth_figures) > 0 else 0

def avg_indicator_after(indicator_series, base_year, country_code, num_years=5, max_year=2021):
  series_country = indicator_series[(indicator_series['Country Code'] == country_code)]
  if np.isnan(base_year): # as sometimes projects are not complete yet
      return np.nan
  
  convert_to_year = lambda i: str(int(base_year) + i) 
  years_for_growth = [convert_to_year(i+1) for i in range(num_years) if int(convert_to_year(i+1)) < max_year]
  growth_figures = series_country.iloc[0][years_for_growth].to_list()
  growth_figures = [gf for gf in growth_figures if not np.isnan(gf)]
  return sum(growth_figures) / len(growth_figures) if len(growth_figures) > 0 else 0

def extract_indicator_prior_project(project, target_col_name, indicator_series, num_years=5, min_year=1960, include_global=True):
  get_indicator_for_code = lambda ccode: avg_indicator_prior(indicator_series, base_year=project["start_year"], country_code=ccode, num_years=num_years, min_year=min_year)
  project[target_col_name] = get_indicator_for_code(project['country_code'])
  if include_global:
    project[f"global_{target_col_name}"] = get_indicator_for_code("WLD")
  return project

def extract_indicator_during_after_project(project, target_col_name, indicator_series, base_year_col, num_years=5, min_year=1960, max_year=2020, include_global=True):
  base_year = project[base_year_col] if int(project[base_year_col]) > min_year else min_year
  get_indicator_avg_for_code = lambda ccode: avg_indicator_after(indicator_series, base_year, country_code=ccode, num_years=num_years, max_year=max_year)
  project[target_col_name] = get_indicator_avg_for_code(project['country_code'])
  if include_global:
    project[f'global_{target_col_name}'] = get_indicator_avg_for_code('WLD')
  return project

def extract_indicator_pre_during_post(project, target_col_base, indicator_series, num_years=5, min_year=1960, max_year=2020, num_years_post=None):
  project = extract_indicator_prior_project(project, target_col_base + "_prior", indicator_series, num_years=num_years, min_year=min_year)
  project = extract_indicator_during_after_project(project, target_col_base + "_during", indicator_series, "start_year", min_year=min_year, max_year=max_year)
  num_years_post = num_years_post or num_years
  project = extract_indicator_during_after_project(project, target_col_base + "_post", indicator_series, "end_year", num_years=num_years_post, min_year=min_year, max_year=max_year)
  return project
