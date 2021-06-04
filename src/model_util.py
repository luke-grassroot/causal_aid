from dowhy import CausalModel
from IPython.display import Image, display

from futil import *

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