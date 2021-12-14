from settings import args_dict, properties
import os

def make_args_list(comp_type: str):
    """Makes a list of strings to print to the argument file"""
    if comp_type in args_dict:
        return [" ".join([args_dict[comp_type]["exe"], property, *pair]) for pair in args_dict[comp_type]["pairs"] for property in properties]
    else:
        raise ValueError("Not valid comp_type={}. Allowed types {}".format(comp_type, ", ".join(args_dict.keys())))
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to arguments file")
    path = parser.parse_args().path
    with open(path, "w") as f:
        #for l in make_args_list("tensorflow") + makes_args_list("deepchem") + make_args_list("sklearn"):
        for l in make_args_list("tensorflow") + make_args_list("sklearn"):
            f.write(l + "\n")
    
    