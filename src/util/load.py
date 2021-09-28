import pandas as pd
import numpy as np

# some helper methods mostly to improve legibility in main notebooks
def load_wdi(data_path='../data/wdi/WDIData.csv', series_path='../data/wdi/WDISeries.csv'):
  return pd.read_csv(data_path, low_memory=False), pd.read_csv(series_path, low_memory=False)

def project_size_USD_calculated(row):
    donor = row['donor_name']
    if donor == "AsianDB":
        return row['asdb_approvedamount'] * 1e6
    elif donor == "DFID":
        return row['dfid_projectbudgetcurrent'] * 1.51
    elif donor == "GFATM":
        return row['gfatm_projectdisbconst_amount']
    elif donor == "GiZ":
        return row['giz_projectsize'] * 1306.5
    elif donor == "IFAD":
        return row['ifad_projectsize'] * 1e6
    elif donor == "JICA":
        return row['jica_projectsize'] * 10687
    elif donor == "KfW":
        return row['kfw_projectsize'] * 1.28
    elif donor == 'WB':
        return row['wb_lendingproject_cost']
    else:
        return 0

def load_projects(data_path="../data/aid_projects.csv", convert_dt=True, merge_purpose=True):
  df = pd.read_csv(data_path, low_memory=False)

  if convert_dt:
    df['start_dt'] = pd.to_datetime(df['start_date'], format='%d%b%Y', errors='coerce')
    df['start_year'] = df['start_dt'].dt.year
    df['completion_dt'] = pd.to_datetime(df['completion_date'], format='%d%b%Y', errors='coerce')
    df['end_year'] = df['completion_dt'].dt.year

  if merge_purpose:
    sector_mapping_table = pd.read_csv('../data/sector_mapping.csv')
    df = df.merge(
        sector_mapping_table[['mmg_purpose_sector', 'sector']], 
        left_on='mmg_purpose_sector', 
        right_on='mmg_purpose_sector',
        how='left'
    )

  df['project_size_USD_calculated'] = df.apply(project_size_USD_calculated, axis=1)
  
  return df

def narrow_convert_project_data(df, relevant_cols, start_date_col='start_date', end_date_col='completion_date', drop_na_end_date=True):
  df = df[relevant_cols]
  df['start_dt'] = pd.to_datetime(df[start_date_col], format='%d%b%Y', errors='coerce')
  df['start_year'] = df['start_dt'].dt.year
  df['completion_dt'] = pd.to_datetime(df[end_date_col], format='%d%b%Y', errors='coerce')
  df['end_year'] = df['completion_dt'].dt.year
  if drop_na_end_date:
    df = df.dropna(subset=['end_year'])
  return df

def extract_wb_projects(project_df):
  wb_df = project_df[project_df.donor_name == 'WB']
  wb_df['start_date'] = pd.to_datetime(wb_df.start_date)
  wb_df['created_year'] = wb_df.start_date.dt.year



def assemble_sector_ratings(project_df, sector):
  sector_df = project_df[project_df.sector == sector]

  end_years_ids = sector_df.groupby(['country_code', 'end_year', 'ppd_project_id'], as_index=False).agg(
    mean_proj_rating=('six_overall_rating', 'mean'),
    total_proj_size=('project_size_USD_calculated', 'sum'),
    project_rating=('six_overall_rating', 'mean')
  )

  def wm(series):
    project_sizes = end_years_ids.iloc[series.index]['total_proj_size']
    if np.any(project_sizes == 0):
        return np.average(series)
    else:
        return np.average(series, weights=end_years_ids.iloc[series.index]['total_proj_size'])

  treatment_df = end_years_ids.groupby(['country_code', 'end_year'], as_index=False).agg(
    num_projs=('ppd_project_id', 'nunique'),
    total_proj_size=('total_proj_size', 'sum'),
    w_avg_rating=('project_rating', wm),
    min_rating=('project_rating', min),
    max_rating=('project_rating', max)
  )

  return treatment_df
  