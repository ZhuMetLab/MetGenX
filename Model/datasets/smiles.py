# !/usr/bin/env python
# -*-coding:utf-8 -*-

from rdkit import Chem
from rdkit.Chem import MolStandardize
class SMILESStandarder(object):
    def __init__(self):
        self.normizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
    def Standard(self, smiles, isomericSmiles=False, canonical=True, kekuleSmiles=True, CanonicalTautomer=False):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = self.normizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            if CanonicalTautomer:
                mol_tautomer = Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)
                inchikey1 = Chem.MolToInchi(mol_tautomer)
                inchikey2 = Chem.MolToInchi(mol)
                if inchikey2==inchikey1:
                    mol = mol_tautomer
                # else:
                #     print(f"Warning for smiles {smiles}")
            Canoical_smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles, canonical=canonical, kekuleSmiles=kekuleSmiles)
            return Canoical_smiles
        except:
            return None

