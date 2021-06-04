from IPython.display import Image, display

from dowhy import CausalModel

from futil import *

sectoral_effect_graph_binary = """digraph { 
    "Govt-Quality"[latent];
    project_completed_year [exposure, label="project_completed"];
    sector_delta_prior [adjusted, label="sector_outcome_growth_prior"];
    sector_delta_post [outcome, label=sector_outcome_growth_lagged];
    most_recent_govt_rating [label="WBGovtRating"];
    "Govt-Quality" -> project_completed_year;
    "Govt-Quality" -> sector_delta_prior;
    "Govt-Quality" -> most_recent_govt_rating;
    project_completed_year -> sector_delta_post;
    sector_delta_prior -> sector_delta_post;
    most_recent_govt_rating -> project_completed_year;
    }"""

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
    
def run_sector_experiment(pdf, sector, sector_indicators, year_range=None, 
                            refuters_to_run=[], interpolate=False, verbose=False, persist_lag_table=True):
  year_range = year_range if year_range is not None else range(int(pdf.start_year.min()), int(pdf.start_year.max()))
  table_path = f'../data/transformed_data/{sector.lower()}_df.csv'

  df, tdf = assemble_sectoral_df(pdf, sector_indicators, year_range, persisted_lag_table=table_path)

  if interpolate:
    print('Add outcome interpolation here')

  print('Constructing or loading sector table')
  print('Columns so far: ', df.columns)
  df = construct_sector_lag_table(df, tdf, pdf, sector_indicators, sector)

  if 'most_recent_govt_rating' not in df:
    df['most_recent_govt_rating'] = df.apply(lambda row: find_latest_govt_rating(pdf, row['country'], row['year']), axis=1)
  
  if persist_lag_table:
    df.to_csv(table_path, index=False)

  df['sector_delta_prior'] = df[f'{sector.lower()}_lag_-4_growth']
  df['sector_delta_post'] = df[f'{sector.lower()}_lag_4_growth']

  causal_columns = ['sector_delta_prior', 'most_recent_govt_rating', 'project_completed_year', 'sector_delta_post']

  causal_df = df[causal_columns]
  pre_drop_N = len(causal_df)
  causal_df = causal_df.dropna()
  post_drop_N = len(causal_df)

  print("N pre NA drop: ", pre_drop_N, " and post: ", post_drop_N)

  model = CausalModel(
    data=causal_df,
    graph=sectoral_effect_graph_binary.replace("\n", " "),
    treatment="project_completed_year",
    outcome="sector_delta_post"
  )

  if verbose:
    model.view_model()
    display(Image(filename="causal_model.png"))

  identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
  estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression",
                                 target_units="ate", test_significance=True)

  refuter_results = []
  if len(refuters_to_run) > 0:
    run_refuter = lambda refuter_name, refuter_params: model.refute_estimate(identified_estimand, estimate, method_name=refuter_name, **refuter_params)
    refuter_results = [run_refuter(refuter[0], refuter[1]) for refuter in refuters_to_run]

  return {
    'causal_model': model, 
    'identified_estimand': identified_estimand, 
    'estimates': [estimate],
    'refutations': refuter_results, 
    'causal_df': causal_df, 
    'reference_df': df
  }