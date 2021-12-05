"""Script to convert the raw into single CSV file and remove unwanted entries"""
import pandas as pd
from rdkit import Chem


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
    mdf = pd.read_csv('../../data/raw/goodstructures_smiles_noLin5.csv', sep=' ')
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


if __name__ == "__main__":
    calculated_df = pd.read_csv('../../data/raw/outputDFT.csv', index_col=0)
    structural_df = pd.read_csv('../../data/raw/structural_features.csv', index_col=0)
    smiles_df = get_valid_molecule_dataframe()
    res = join_all_dataframes(calculated_df, structural_df, smiles_df)
    res.to_csv("../../data/all_data.csv")