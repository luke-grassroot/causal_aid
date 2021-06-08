import pandas as pd

# some helper methods mostly to improve legibility in main notebooks
def load_wdi(data_path='../data/wdi/WDIData.csv', series_path='../data/wdi/WDISeries.csv'):
  return pd.read_csv(data_path, low_memory=False), pd.read_csv(series_path, low_memory=False)

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
