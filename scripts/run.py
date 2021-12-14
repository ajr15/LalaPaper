import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models import BaseEstimator
from src.core.representations import Representation
from src.commons import make_data, split_train_test, run_bayes_cv, analyze_model
import settings

def main_run(model: BaseEstimator, rep: Representation, property: str):
    """Main run function for model, representation and property. 
    Makes hyperparameter optimization, trains model on final training set and saves all required data."""
    # making data
    if property in settings.properties:
        X, y = make_data(property.lower(), rep)
    # error if not valid property
    else:
        raise ValueError("Unrecognized property {}".format(property))
    # splits to train / test
    x_train, x_test, y_train, y_test = split_train_test(X, y, settings.test_size)
    # running CV
    opt_model = run_bayes_cv(model, 
                                settings.model_search_spaces[model.name],
                                x_train, 
                                y_train, 
                                settings.cv_number, 
                                settings.cv_n_iter,
                                settings.cv_njobs)
    # making appropriate results directory
    prop_dir = os.path.join(settings.parent_res_dir, property)
    if not os.path.isdir(prop_dir):
        os.mkdir(prop_dir)
    res_dir = os.path.join(prop_dir, "{}_{}".format(model.name, rep.name))
    # estimating performance & saving model
    analyze_model(opt_model, x_train, y_train, x_test, y_test, res_dir, **settings.fit_plot_kwargs)


def print_msg(msg: str):
    l = max(int(round(len(msg) * 1.5)), 50)
    pre_space = int(round(l / 2 - len(msg) / 2))
    print("=" * l)
    print(" " * pre_space + msg)
    print("=" * l)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single model-representation pair")
    parser.add_argument("model", type=str, help="Name of the model. Allowed values {}".format(", ".join(settings.models_dict.keys())))
    parser.add_argument("representation", type=str, help="Name of the representaion. Allowed values {}".format(", ".join(settings.reps_dict.keys())))
    parser.add_argument("property", type=str, help="Name of the property. Allowed values {}".format(", ".join(settings.properties)))
    args = parser.parse_args()
    print_msg("Running for model {} with representation {} for {} property".format(args.model, args.representation, args.property))
    model = settings.models_dict[args.model]
    rep = settings.reps_dict[args.representation]
    try:
        main_run(model, rep, args.property)
        print_msg("COMPUTATION ENDED SUCCESSFULLY")
    except Exception as err:
        print_msg("COMPUTATION ENDED WITH AN ERROR")
        raise err