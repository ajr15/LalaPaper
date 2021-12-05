"""Settings for all calculations to come.
Contains
    - hyperparameter search space for BayesCV
    - Model - representation pairs
    - properties - list of properties for calculations"""
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import models
from src.core import representations as reps
from skopt.space import Real, Integer, Categorical


# list of (model, representation) pairs, to make data easily
tf_run_model_representation_pairs = [
    (models.RNN, reps.LalaOneHot()),
    (models.RNN, reps.SmilesOneHot())
]

dc_run_model_representation_pairs = [(models.GraphConv, reps.MolConvRepresentation())]

sk_run_model_representation_pairs = [
    (models.RfModel, reps.LalaFeatures()),
    (models.CustodiRfModel, reps.LalaString(padd=True)),
    (models.CustodiRfModel, reps.SmilesString(padd=True)),
    (models.KrrModel, reps.LalaFeatures()),
    (models.CustodiKrrModel, reps.LalaString(padd=True)),
    (models.CustodiKrrModel, reps.SmilesString(padd=True)),
    (models.CustodiModel, reps.LalaString(padd=False)),
    (models.CustodiModel, reps.SmilesString(padd=False))
]

models_dict = {
    "rnn": models.RNN,
    "custodi_model": models.CustodiModel,
    "graph_conv": models.GraphConv,
    "krr_model": models.KrrModel,
    "rf_model": models.RfModel,
    "krr_model_custodi_rep": models.CustodiKrrModel,
    "rf_model_custodi_rep": models.CustodiRfModel
    }

reps_dict = {
        "lala_one_hot_rep": reps.LalaOneHot(),
        "smiles_one_hot_rep": reps.SmilesOneHot(),
        "mol_conv_rep": reps.MolConvRepresentation(),
        "lala_features_rep": reps.LalaFeatures(),
        "smiles_str_rep": reps.SmilesString(),
        "smiles_str_rep_padd": reps.SmilesString(padd=True),
        "lala_str_rep": reps.LalaString(),
        "lala_str_rep_padd": reps.LalaString(padd=True)
}

# list of model hyperparameter search spaces
model_search_spaces = {
    "rnn": {
            "learning_rate": Real(1e-6, 1e-1),
            "activation": Categorical(["tanh", "relu"]),
            "dropout_rate": Real(0, 1)
           },
    "custodi_model": {
            "degree": Integer(1, 6),
            "alpha": Real(1e-5, 1e2),
            "max_iter": int(1e5)
           },
    "graph_conv": {
            "n_filters": Integer(64, 256),
            "n_fully_connected_nodes": Integer(64, 256),
            "learning_rate": Real(1e-6, 1e-1),
            "batch_size": Integer(64, 256),
            "nb_epochs": int(100)
           },
    "krr_model": {
            "alpha": Real(1e-5, 1e2),
            "kernel": Categorical(["linear", "rbf"]),
           },
    "rf_model": {
            "n_estimators": int(500)
            },
    "krr_model_custodi_rep": {
            "degree": Integer(1, 6),
            "custodi_alpha": Real(1e-5, 1e2),
            "max_iter": int(1e5),
            "alpha": Real(1e-5, 1e2),
            "kernel": Categorical(["rbf"]),
           },
    "rf_model_custodi_rep": {
            "n_estimators": int(500)
           }
    }

# list of properties to run for
properties = ["HOMO_eV",
              "LUMO_eV",
              "GAP_eV",
              "Dipmom_Debye",
              "Etot_eV",
              "Etot_pos_eV",
              "Etot_neg_eV",
              "aEA_eV",
              "aIP_eV"]
properties = [s.lower() for s in properties]

# general running parameters
cv_number = 5
cv_n_iter = 10
fit_plot_kwargs = {"alpha": 0.2}
parent_res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
test_size = 1000 # a thousand molecules test set