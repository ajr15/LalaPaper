import pandas as pd
import os

model_rep_pairs = [
          ("rf_model", "lala_features_rep"),
          ("rf_model_custodi_rep", "lala_str_rep"),
          ("rf_model_custodi_rep", "smiles_str_rep"),
          ("krr_model","lala_features_rep"),
          ("krr_model_custodi_rep", "lala_str_rep"),
          ("krr_model_custodi_rep", "smiles_str_rep"),
          ("custodi_model", "lala_str_rep"),
          ("custodi_model", "smiles_str_rep"),
          ("rnn", "lala_one_hot_rep"), 
          ("rnn", "smiles_one_hot_rep"),
          ("graph_conv", "mol_conv_rep")
        ]

def parse_results_dir_str(path):
    """Method to parse the results directory path into property, model and representation.
    RETURNS: dictionary with property, model name and represntation name"""
    property = os.path.split(os.path.dirname(path))[-1]
    for model, rep in model_rep_pairs:
        if model in path and rep in path:
            return {"property": property, "model": model, "representation": rep}
            
def unite_all_results_to_csv(parent_res_dir, error_measure):
    main_df = pd.DataFrame()
    for prop_dir in os.listdir(parent_res_dir):
        if not "test" in prop_dir and not ".csv" in prop_dir:
            for res_dir in os.listdir(os.path.join(parent_res_dir, prop_dir)):
                print(os.path.join(parent_res_dir, prop_dir, res_dir))
                res_df = pd.read_csv(os.path.join(parent_res_dir, prop_dir, res_dir, "error_metrics.csv"), index_col="Unnamed: 0")
                res_d = res_df.loc[error_measure, :].to_dict()
                res_d.update(parse_results_dir_str(os.path.join(parent_res_dir, prop_dir, res_dir)))
                main_df = main_df.append(res_d, ignore_index=True)
    test_size = os.path.split(parent_res_dir)[-1]
    main_df.to_csv(os.path.join("../results/models", "prased_results", "{}_test_{}_results.csv".format(test_size, error_measure)))
    
if __name__ == "__main__":
    # parsing results
    for parent in os.listdir("../results/models"):
        unite_all_results_to_csv(os.path.join("../results/models", parent), "rmse")
        unite_all_results_to_csv(os.path.join("../results/models", parent), "mae")
    # calculating metrics for data
    data = pd.read_csv("../data/all_data.csv")
    data.describe().to_csv(os.path.join("../results/models", "prased_results", "data_descrps.csv"))
    