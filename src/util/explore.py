import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from os.path import isfile

# utilities for extracting prior and post on indicators
def extract_indicators(wdi_df, indicator_codes):
  return wdi_df[wdi_df['Indicator Code'].isin(indicator_codes)]

def lag_variable_simple(data, col, num_years):
    data = data.sort_values(['country', 'year'])
    data[f"{col}_lag{num_years}"] = data.groupby('country')[col].shift(num_years)
    return data

def rolling_country_agg(data, col, num_years, agg_function):
    data = data.sort_values(['country', 'year'])
    t_l = lambda x: x.rolling(num_years, 1).max() if agg_function == 'max' else x.rolling(num_years, 1).mean()
    return data.groupby('country')[col].transform(t_l)

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

def extract_indic_name_definition(series_df, string_fragment, columns=['Series Code', 'Indicator Name', 'Long definition']):
    xdf = series_df[series_df['Indicator Name'].str.contains(string_fragment)]
    return xdf[columns]

def calculate_ratio_na(wdi_df, indicator_code, country_codes, earliest_year=1970):
    indic_extract = extract_indicators(wdi_df, [indicator_code])
    tdf = indic_extract.drop(columns=['Indicator Name', 'Indicator Code', 'Unnamed: 65'], errors='ignore').melt(id_vars=['Country Code', 'Country Name'])
    tdf['year'] = pd.to_datetime(tdf['variable']).dt.year
    tdf = tdf[tdf['year'] > earliest_year]
    tdf = tdf[tdf['Country Code'].isin(country_codes)]
    return tdf.value.isna().sum() / len(tdf)

def get_countries_with_active_project(pdf, year, year_lag=5, in_sector=None):
    ref_year = datetime(year, 1, 1)
    
    active_mask = pdf.start_dt < ref_year
    lag_mask = (pdf.completion_dt + timedelta(days=(year_lag*365))) > ref_year
    active_plus_lag_mask = active_mask & lag_mask
    sector_mask = pdf.sector == in_sector
    
    final_mask = (active_plus_lag_mask & sector_mask) if in_sector is not None else active_plus_lag_mask
    
    return pdf[final_mask].country_code.to_list()

def count_untreated_countries_in_window(pdf, year_range, year_lag=5, in_sector=None):
    total_countries = pdf.country_code.nunique()
    countries_with_active_projects = []

    for year in year_range:
        ref_year = datetime(year, 1, 1)
        
        active_mask = pdf.start_dt < ref_year
        lag_mask = (pdf.completion_dt + timedelta(days=(5*365))) > ref_year
        active_plus_lag_mask = active_mask & lag_mask
        sector_mask = pdf.sector == in_sector
        
        final_mask = (active_plus_lag_mask & sector_mask) if in_sector is not None else active_plus_lag_mask
        
        countries_with_active_projects.append(pdf[final_mask].country_code.nunique())
    
    prop_countries_untreated = [1 - count / total_countries for count in countries_with_active_projects]
    return prop_countries_untreated
  
def get_avg_indicator_with_lag(tdf, country, year, indicators, lag):
    country_df = tdf[(tdf['Country Code'] == country) & tdf['Indicator Code'].isin(indicators)]
    max_year = country_df['Year'].max()
    
    if max_year < year + lag:
        return np.nan, 0
    
    ref_year = country_df[(country_df['Year'] == year) & (country_df['value'].notna())]
    ref_year_indicators = set(ref_year['Indicator Code'].unique())
    
    if (len(ref_year_indicators)) == 0:
        return np.nan, 0

    future_year = country_df[(country_df['Year'] == year + lag) & (country_df['value'].notna())]
    future_year_indicators = set(future_year['Indicator Code'].unique())
    
    if (len(future_year_indicators)) == 0:
        return np.nan, 0
    
    both_present_indic = ref_year_indicators & future_year_indicators
    
    if len(both_present_indic) == 0:
        return np.nan, 0
    
    extract_value = lambda indicator, ydf: ydf[ydf['Indicator Code'] == indicator].iloc[0].value      
    growth_values = [extract_value(indicator, future_year) / extract_value(indicator, ref_year) for indicator in both_present_indic]
    
    return sum(growth_values) / len(growth_values), len(both_present_indic)

def calc_sector_controls(pdf, sector, start_year, year_range):
    assemble_sector_dict = lambda sector: dict(zip(year_range, count_untreated_countries_in_window(pdf, year_range, year_lag=5, in_sector=sector)))
    sector_post_year = [value for key, value in assemble_sector_dict(sector).items() if key >= start_year]
    mean_post_year = sum(sector_post_year) / len(sector_post_year)
    return mean_post_year

# assembling or loading the (quite heavy) sector outcome lag tables
def melt_interpolate_df(tdf, drop_columns=None, interpolate_limit=None):
    if drop_columns is not None:
      tdf = tdf.drop(columns=drop_columns)

    if interpolate_limit is not None:
      tdf = tdf.set_index(['Country Code', 'Indicator Code'])
      tdf = tdf.interpolate(axis=1, method='linear', limit=interpolate_limit) # only linear allowed on multi index anyway, but being explicit
      tdf = tdf.reset_index()

    tdf = tdf.melt(
        id_vars=['Country Code', 'Indicator Code'],
        var_name="Year"
    )
    tdf['Year'] = tdf['Year'].astype(int)
    return tdf

def assemble_sectoral_df(pdf, sector_indicators, year_range, persisted_lag_table=None, interpolate_limit=None):
    # first, assemble DF from cross product of country codes and all years
    all_countries = pdf.country_code.unique()
    df = pd.DataFrame(index = pd.MultiIndex.from_product([year_range, all_countries], names=["year", "country"])).reset_index()

    # second, look up education indicators at years + lag, after interpolating
    mdf = pd.read_csv('../data/MDGData.csv')
    edf = mdf[mdf['Indicator Code'].isin(sector_indicators)]
    tdf = edf.drop(
        columns=["Country Name", "Indicator Name", "Unnamed: 30"]
    )

    if interpolate_limit is not None:
      tdf = tdf.set_index(['Country Code', 'Indicator Code'])
      tdf = tdf.interpolate(axis=1, method='linear', limit=interpolate_limit) # only linear allowed on multi index anyway, but being explicit
      tdf = tdf.reset_index()

    tdf = tdf.melt(
        id_vars=['Country Code', 'Indicator Code'],
        var_name="Year"
    )
    tdf['Year'] = tdf['Year'].astype(int)
    
    # since this is a heavy operation, rather just reload, when available
    if persisted_lag_table is not None and isfile(persisted_lag_table):
        df = pd.read_csv(persisted_lag_table)
    
    return df, tdf

def is_project_complete_year(year, country, sp_df):
  country_proj = sp_df[sp_df['country_code'] == country].end_year.to_list()
  return bool(year in country_proj)

def assemble_sector_proj_df(project_df, sector, consolidated_df=None, ppd_country_col='ppd_countrycode'):
  sp_df = project_df[project_df.sector == sector]
  start_years = sp_df.groupby('country_code')['start_year'].apply(set).to_dict()
  
  if consolidated_df is not None:
    consolidated_df['project_start_year'] = consolidated_df.apply(
        lambda row: row[ppd_country_col] in start_years and row['year'] in start_years[row[ppd_country_col]], 
    axis=1)
    consolidated_df['project_completed_year'] = consolidated_df.apply(lambda row: is_project_complete_year(row['year'], row[ppd_country_col], sp_df), axis=1)

  return sp_df, consolidated_df

def construct_sector_lag_table(df, time_df, project_df, sector_indicators, sector, 
                                lag_range=range(1, 10), prior_growth_offset=4, 
                                force_recalculation=False, persist_if_modified=False, persist_file_name=None):
    
    col_name = lambda lag: sector.lower() + '_lag_' + str(lag)

    modified_df = False

    # first add the lags
    def add_edu_lag_and_count(row, lag):
      lag_avg, lag_count = get_avg_indicator_with_lag(time_df, row['country'], row['year'], sector_indicators, lag)
      row[col_name(lag) + '_growth'] = lag_avg
      row[col_name(lag) + '_count'] = lag_count
      return row

    for lag in lag_range:
      if force_recalculation or col_name(lag) + '_growth' not in df.columns:
        modified_df = True
        df = df.apply(lambda row: add_edu_lag_and_count(row, lag), axis=1)

    if force_recalculation or col_name(-prior_growth_offset) + '_growth' not in df.columns:
      modified_df = True
      df = df.apply(lambda row: add_edu_lag_and_count(row, -4), axis=1)

    if len(df) == 0:
      raise Exception('Error! Dataframe is empty')

    # then add project completion (or not)
    if 'project_completed_year' not in df.columns:
      sp_df = project_df[project_df.sector == sector]
      start_years = sp_df.groupby('country_code')['start_year'].apply(set).to_dict()
      df['project_completed_year'] = df.apply(
        lambda row: row['country'] in start_years and row['year'] in start_years[row['country']], axis=1)

    if modified_df and persist_if_modified:
      persist_file = persist_file_name or f'{sector.lower()}_df.csv'
      df.to_csv(f'../data/transformed_data/f{persist_file}', index=False)

    return df
    