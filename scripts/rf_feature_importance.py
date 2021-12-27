import pandas as pd
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models import RfModel
import settings


def load_model_for_prop(property, test_size):
    model_dir = "../results/models/{}/{}/rf_model_lala_features_rep/model".format(test_size, property)
    model = RfModel.load(model_dir)
    return model

    
if __name__ == "__main__":
    feature_names = ["n_branches", 
                      "longest_a", 
                      "longest_l", 
                      "longest_l_degeneracy",
                      "second_longest_l",
                      "ratio_l",
                      "n_lal",
                      "n_rings"]
    importance_df = pd.DataFrame()
    for property in settings.properties:
        model = load_model_for_prop(property, 1000)
        importances = {name: im for name, im in zip(feature_names, model.feature_importances_)}
        importances["property"] = property
        importance_df = importance_df.append(importances, ignore_index=True)
    importance_df.to_csv("../results/feature_importance/rf_feature_importances.csv")
        