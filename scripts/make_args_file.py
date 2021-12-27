from settings import args_dict, properties
import os

def make_args_list(comp_type: str):
    """Makes a list of strings to print to the argument file"""
    if comp_type in args_dict:
        args = []
        for test_size in [1000, 2500, 5000, 7600]:
            for property in properties:
                for pair in args_dict[comp_type]["pairs"]:
                    model = pair[0]
                    rep = pair[1] if not "padd" in pair[1] else pair[1][:-5]
                    print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                    if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                        args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
        return args
    else:
        raise ValueError("Not valid comp_type={}. Allowed types {}".format(comp_type, ", ".join(args_dict.keys())))
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to arguments file")
    path = parser.parse_args().path
    with open(path, "w") as f:
        #for l in make_args_list("tensorflow") + makes_args_list("deepchem") + make_args_list("sklearn"):
        for l in make_args_list("tensorflow") + make_args_list("sklearn") + make_args_list("deepchem"):
            f.write(l + "\n")
    
    