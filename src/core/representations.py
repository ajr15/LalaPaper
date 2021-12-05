"""File to contain representations"""

from abc import ABC, abstractclassmethod
from typing import List
import numpy as np

try:
    from rdkit import Chem
    import deepchem as dc
except ImportError:
    pass

from ..commons import read_raw_columns

class Representation (ABC):

    @abstractclassmethod
    def represent(self, molecule_ids: List[str]):
        """Method to make representation for molecules given their IDs in the dataset"""
        pass


class OneHot (Representation):
    """Generic one-hot tokenization (includes padding of strings of different lengths)"""

    def __init__(self, strings):
        self.tokenization_dict = self.set_tokenization_dict(strings)
        self.max_length = max([len(s) for s in strings])

    @abstractclassmethod
    def set_tokenization_dict(strings: List[str]):
        """Method to generate standard word tokenization dictionary"""
        d = {}
        for char in np.ravel(strings):
            if char not in d.keys():
                d[char] = len(d.keys())
        return d


    def tokenize_string(self, string):
        """Method to tokenize a string"""
        tok = []
        for s in string:
            v = np.zeros(len(self.tokenization_dict) + 2)
            v[self.tokenization_dict[s]] = 1
            tok.append(v)
        padd_vec = np.zeros(len(self.tokenization_dict) + 2)
        padd_vec[-1] = 1
        while len(tok) < self.max_length:
            tok.append(padd_vec)
        return tok


    def represent(self, molecule_id: str):
        return super().represent(molecule_id)


class LalaOneHot (OneHot):
    """LALA based one-hot"""

    __name__ == "lala_one_hot_rep"


    def __init__(self):
        strings = read_raw_columns(["annulation"]).values
        super().__init__(strings)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["annulation"]).loc[molecule_ids, :]
        return [self.tokenize_string(s) for s in strings]


class SmilesOneHot (OneHot):
    """SMILES based one-hot"""

    __name__ == "smiles_one_hot_rep"


    def __init__(self):
        strings = read_raw_columns(["smiles"]).values
        super().__init__(strings)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["smiles"]).loc[molecule_ids, :]
        return [self.tokenize_string(s) for s in strings]


class MolConvRepresentation (Representation):
    """Representation as a deepchem.ConvMol for graph convolutions data"""

    __name__ == "mol_conv_rep"
    
    def tokenize_smiles(self, smiles: str):
        rdmol = Chem.MolFromSmiles(smiles, sanitize=True)
        featurizer = dc.feat.ConvMolFeaturizer()
        return featurizer.featurize(rdmol)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["smiles"]).loc[molecule_ids, :]
        return [self.tokenize_smiles(s) for s in strings]


class LalaFeatures (Representation):
    """Representation as the LALA feature vector"""

    __name__ == "lala_features_rep"
    
    def represent(self, molecule_ids: List[str]):
        return read_raw_columns(["smn_branches", 
                                    "longest_A", 
                                    "longest_L", 
                                    "longest_L_degeneracy",
                                    "second_longest_L",
                                    "ratio_L",
                                    "n_LAL",
                                    "n_ringsiles"]).loc[molecule_ids, :]
        

def _padd_strings(strings: List[str]):
    """private method to padd list of strings. returns padded stings"""
    max_l = max([len(s) for s in strings])
    for i in range(len(strings)):
        while len(strings[i]) < max_l:
            strings[i] += " "
    return strings


class SmilesString (Representation):
    """Representation that returns the SMILES string of the molecules as representation"""

    __name__ == "smiles_str_rep"

    def __init__(self, padd=False) -> None:
        self.padd = padd


    def represent(self, molecule_ids: List[str]):
        if self.padd:
            return _padd_strings(read_raw_columns(["smiles"]).loc[molecule_ids, :])
        else:
            return read_raw_columns(["smiles"]).loc[molecule_ids, :]


class LalaString (Representation):
    """Representation that returns the LALA string of the molecules as representation"""

    __name__ == "lala_str_rep"

    
    def __init__(self, padd=False) -> None:
        self.padd = padd


    def represent(self, molecule_ids: List[str]):
        if self.padd:
            return _padd_strings(read_raw_columns(["annulation"]).loc[molecule_ids, :])
        else:
            return read_raw_columns(["annulation"]).loc[molecule_ids, :]