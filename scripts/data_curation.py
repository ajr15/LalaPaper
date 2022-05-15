"""Script to convert the raw into single CSV file and remove unwanted entries"""
import pandas as pd
from rdkit import Chem
from matplotlib import image, pyplot as plt
import os


def is_rdkit_readable(smiles: str) -> bool:
    """method to check if a smiles string is rdkit readable"""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return False
    else:
        return True


def get_valid_molecule_dataframe():
    """Method to get the rdkit readable molecules with at least 3 benzene rings. Returns a dataframe with valid molecule IDs and their SMILES"""
    # reading data
    mdf = pd.read_csv('../data/raw/goodstructures_smiles_noLin5.csv', sep=' ')
    print("number of molecules before filtration:", len(mdf))
    # removing all molecules with less than 3 rings
    mdf = mdf[~mdf.molecule_id.str.contains('c46h26')].reset_index(drop=True)
    mdf = mdf[~mdf.molecule_id.str.contains('c6h6')].reset_index(drop=True)
    mdf = mdf[~mdf.molecule_id.str.contains('c10h8')].reset_index(drop=True)
    mdf = mdf.set_index('molecule_id')
    print("Number of molecules after number of rings filtration:", len(mdf))
    # finding rdkit readable molecules
    idxs = []
    for i, smiles in enumerate(mdf["smiles"]):
        if is_rdkit_readable(smiles):
            idxs.append(i)
    # returning correct dataframe
    print("number of molecules after rdkit filtration:", len(idxs))
    return mdf.iloc[idxs, :]


def join_all_dataframes(calculated_df, stractural_df, smiles_df):
    """joining all dataframes based on their mutual indicis"""
    # adding structural data
    res = pd.merge(smiles_df, stractural_df, left_index=True, right_index=True)
    # adding calculated data
    res = pd.merge(res, calculated_df, left_index=True, right_index=True)
    return res

def get_unique_compositions(molecule_ids):
    """Method to get all the unique compositions (CxHy) from the molecule ids"""
    comps = set()
    for s in molecule_ids:
        comp = s.split("_")[1]
        comps.add(comp)
    return list(comps)

def calc_atom_energy(comp):
    """Method to calculate the energy of C and H in the gas phase, for atomization energy calculation. given composition (CxHy)"""
    carbon_gas_energy_ev = -840
    hydrogen_gas_energy_ev = -210
    num_c = int(comp.split("h")[0][1:])
    num_h = int(comp.split("h")[-1])
    return num_c * carbon_gas_energy_ev + num_h * hydrogen_gas_energy_ev, 1

def calculate_relative_energy(energy_df: pd.DataFrame):
    """Method to calculate the relative energy of the molecules in the dataset (energy = min(isomer energy) - e_isomer)"""
    compositions = get_unique_compositions(energy_df.index)
    res = pd.DataFrame()
    for comp in compositions:
        res = res.append(energy_df[energy_df.index.str.contains(comp)] - energy_df[energy_df.index.str.contains(comp)].min())
    res.columns = ["rel_" + col for col in res.columns]
    return res

def calculate_atomization_energy(energy_df: pd.DataFrame):
    """Method to calculate the atomization energy of the molecules in the dataset"""
    compositions = get_unique_compositions(energy_df.index)
    res = pd.DataFrame()
    for comp in compositions:
        energy, factor = calc_atom_energy(comp)
        res = res.append((energy_df[energy_df.index.str.contains(comp)] - energy) / factor)
    res.columns = ["atom_" + col for col in res.columns]
    return res

def plot_correlation(df, prop1, prop2, image_path):
    print("plotting {} vs. {}".format(prop1, prop2))
    plt.figure()
    plt.scatter(df[prop1], df[prop2], color="black", alpha=0.5)
    plt.xlabel(prop1)
    plt.ylabel(prop2)
    plt.savefig(image_path)
    plt.close()


def plot_hist(df, prop, image_path):
    print("plotting {} histogram".format(prop))
    plt.figure()
    plt.hist(df[prop], color="black", bins=100)
    plt.title(prop)
    plt.savefig(image_path)
    plt.close()


def make_plots(df):
    im_dir = "../results/stats"
    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)
    for i, col1 in enumerate(df.columns):
        plot_hist(df, col1, os.path.join(im_dir, "hists/{}.png".format(col1)))
        for col2 in df.columns[(i + 1):]:
            plot_correlation(df, col1, col2, os.path.join(im_dir, "corr/{}_vs_{}.png".format(col1, col2)))


def parse_old_raw():
    calculated_df = pd.read_csv('../data/raw/outputDFT.csv', index_col=0)
    structural_df = pd.read_csv('../data/raw/structural_features.csv', index_col=0)
    smiles_df = get_valid_molecule_dataframe()
    res = join_all_dataframes(calculated_df, structural_df, smiles_df)
    # adding relative energy
    rel_e = calculate_relative_energy(res.loc[:, ["Etot_eV", "Etot_pos_eV", "Etot_neg_eV"]])
    atom_e = calculate_atomization_energy(res.loc[:, ["Etot_eV", "Etot_pos_eV", "Etot_neg_eV"]])
    res = pd.merge(res, rel_e, left_index=True, right_index=True)
    res = pd.merge(res, atom_e, left_index=True, right_index=True)
    make_plots(res.iloc[:, 3:])
    res.to_csv("../data/all_data.csv")
    

def parse_new_raw():
    # reading data from file
    raw_df = pd.read_csv("../data/raw/raw_data_24042022.csv")
    # dropping irrelevant values
    raw_df = raw_df.drop(columns=["Etot_pos_eV", "Etot_neg_eV", "Dipmom_Debye", "dispersion_eV", "Etot_SCF_eV"])
    raw_df = raw_df[~raw_df.molecule.str.contains('c46h26')].reset_index(drop=True)
    raw_df = raw_df[~raw_df.molecule.str.contains('c6h6')].reset_index(drop=True)
    raw_df = raw_df[~raw_df.molecule.str.contains('c10h8')].reset_index(drop=True)
    # setting index properly
    raw_df = raw_df.set_index("molecule")
    # adding relative energy column
    rel_e = calculate_relative_energy(raw_df.loc[:, ["Etot_eV"]])
    raw_df = pd.merge(raw_df, rel_e, left_index=True, right_index=True)
    # dropping total energy 
    raw_df = raw_df.drop(columns=["Etot_eV"])
    # saving to new file
    raw_df.to_csv("../data/all_data.csv")
    # visualizing data
    #make_plots(raw_df.iloc[:, 2:])
    
    
if __name__ == "__main__":
    parse_new_raw()