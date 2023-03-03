#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:55:08 2022

@author: zdx
"""
import os
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
from collections import Counter
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem, rdinchi
from moses.metrics import mol_passes_filters, QED, SA, logP
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi

def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments

def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def addID(df, pre):
    l = len(str(len(df)))
    nums = []
    init_str = '0' * l
    
    for i in range(len(df)):
        num_str = str(i)
        num_str_len = len(num_str)
        num_str = init_str[:l-num_str_len] + num_str
        nums.append('_'.join([pre, num_str]))
    return nums


"""
-------------------------------------------------------------------------------
Mol to string
-------------------------------------------------------------------------------
"""
def MOL2SMILES(mol):
    try:
        sm = Chem.MolToSmiles(mol)
        return sm
    except:
        return np.nan

"""
-------------------------------------------------------------------------------
String to mol 
-------------------------------------------------------------------------------
"""
def InChI2MOL(inchi):
    try:
        mol = Chem.inchi.MolFromInchi(inchi)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan
    
def SMILES2MOL(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan

"""
-------------------------------------------------------------------------------
Apply string to mol
-------------------------------------------------------------------------------
"""
def apply_SMILES2MOL(df):
    df['ROMol'] = df['SMILES'].apply(SMILES2MOL)
    return df

def apply_InChI2MOL(df):
    df['ROMol'] = df['InChI'].apply(InChI2MOL)
    return df

"""
-------------------------------------------------------------------------------
apply mol to string
-------------------------------------------------------------------------------
"""
def add_mol(df):
    if 'SMILES' in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
    elif 'InChI' in df.columns:
        df = parallelize_dataframe(df, apply_InChI2MOL)
    return df

def MOL2InChI(mol):
    try:
        inchi, retcode, message, logs, aux = rdinchi.MolToInchi(mol)
        return inchi
    except:
        return np.nan
    
def apply_MOL2InChI(df):
    df['InChI'] = df.ROMol.apply(MOL2InChI)
    return df

def SMILES2mol2InChI(df):
    if 'ROMol' not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
    df = parallelize_dataframe(df, apply_MOL2InChI)
    del df['ROMol']
    return df

def add_inchi(df):
    if 'ROMol' not in df.columns:
        df = add_mol(df)
    df = parallelize_dataframe(df, apply_MOL2InChI)
    del df['ROMol']
    return df

"""
=========================== parallelize apply =================================
"""

def parallelize_dataframe(df, func, **kwargs):
    CPUs = multiprocessing.cpu_count()

    num_partitions = int(CPUs*0.8) # number of partitions to split dataframe
    num_cores = int(CPUs*0.8) # number of cores on your machine

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def fileFormat(file):
    if '.gz' in file:
        file = file.replace('.gz', '')
    format_ = os.path.splitext(os.path.basename(file))[1].split('.')[-1]
    if format_ == 'ism':
        format_ = 'smi'
    return format_

def autoFindsmile(df):
    cols = df.columns
    for col in cols:
        if isinstance(col, str):
            if 'C' in col or 'c' in col:
                if 'M' in col or 'L' in col or 'Z' in col:
                    continue
                try:
                    mol = Chem.MolFromSmiles(df[col][0])
                except:
                    mol = None
                if mol is not None:
                    print(mol)
                    df.rename(columns={col:'SMILES'}, inplace=True)
    return df

"""
-------------------------------------------------------------------------------
Property functions
-------------------------------------------------------------------------------
"""
def judge_whether_has_rings_4(mol):
    r = mol.GetRingInfo()
    if len([x for x in r.AtomRings() if len(x)==4]) > 0:
        return False
    else:
        return True
    
def add_whether_have_4_rings(data):
    """4 rings"""
    data['4rings'] = data['ROMol'].apply(judge_whether_has_rings_4)
    return data

def four_rings_filter(df):
    df = parallelize_dataframe(df, add_whether_have_4_rings)
    df = df[df['4rings']==True]
    del df['4rings']
    return df

def MW(mol):
    try:
        res = Chem.Descriptors.ExactMolWt(mol)
        return res
    except:
        return np.nan

def HBA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBA(mol)
        return res
    except:
        return np.nan

def HBD(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBD(mol)
        return res
    except:
        return np.nan

def TPSA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcTPSA(mol)
        return res
    except:
        return np.nan

def NRB(mol):
    try:
        res =  Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        return res
    except:
        return np.nan
    
def get_num_rings(mol):
    try:
        r = mol.GetRingInfo()
        res = len(r.AtomRings())
        return res
    except:
        return np.nan

def get_num_rings_6(mol):
    try:
        r = mol.GetRingInfo()
        res = len([x for x in r.AtomRings() if len(x) > 6])
        return res
    except:
        return np.nan

def LOGP(mol):
    try:
        res = logP(mol)
        return res
    except:
        return np.nan
    
def MCF(mol):
    """
    Keep molecules whose MCF=True
    MCF=True means toxicity. but toxicity=True is not bad if the patient is dying.
    """
    try:
        res = mol_passes_filters(mol)
        return res
    except:
        return np.nan

def synthesis_availability(mol):
    """
    0-10. smaller, easier to synthezie.
    not very accurate.
    """
    try:
        res = SA(mol)
        return res
    except:
        return np.nan
    
def estimation_drug_likeness(mol):
    """
    0-1. bigger is better.
    """
    try:
        res = QED(mol)
        return res
    except:
        return np.nan

def get_scaffold_mol(mol):
    try: 
        res = MurckoScaffold.GetScaffoldForMol(mol)
        return res
    except:
        return np.nan

def add_atomic_scaffold_mol(df):
    df['atomic_scaffold_mol'] = df.ROMol.apply(get_scaffold_mol)
    return df

def get_scaffold_inchi(mol):
    try: 
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        inchi = MOL2InChI(scaffold_mol)
        return inchi
    except:
        return np.nan

def get_scaffold_smiles(mol):
    try: 
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        smiles = MOL2SMILES(scaffold_mol)
        return smiles
    except:
        return np.nan

def add_scaffold_inchi(df):
    df['scaffold_inchi'] = df.ROMol.apply(get_scaffold_inchi)
    return df

def add_descriptors(df, pro=['MW', 'logP', 'HBA', 'HBD', 
                             'TPSA', 'NRB', 'MCF', 'SA', 'QED',
                             'rings', 'scaffold_inchi', 'scaffold_smiles']):
    if 'MW' in pro:
        df['MW'] = df.ROMol.apply(MW)
    if 'logP' in pro:
        df['logP'] = df.ROMol.apply(LOGP)
    if 'HBA' in pro:
        df['HBA'] = df.ROMol.apply(HBA)
    if 'HBD' in pro:
        df['HBD'] =  df.ROMol.apply(HBD)
    if 'TPSA' in pro:
        df['TPSA'] = df.ROMol.apply(TPSA)
    if 'NRB' in pro:
        df['NRB'] = df.ROMol.apply(NRB)
    if 'MCF' in pro:
        df['MCF'] = df.ROMol.apply(MCF)
    if 'SA' in pro:
        df['SA'] = df.ROMol.apply(synthesis_availability)
    if 'QED' in pro:
        df['QED'] = df.ROMol.apply(estimation_drug_likeness)
    if 'rings' in pro:
        df['rings'] = df.ROMol.apply(get_num_rings)
    if 'scaffold_inchi' in pro:
        df['scaffold_inchi'] = df.ROMol.apply(get_scaffold_inchi)
    if 'scaffold_smiles' in pro:
        df['scaffold_smiles'] = df.ROMol.apply(get_scaffold_smiles)
    return df


def validity_filter(df):
    print("Start to remove invalid SMILES...")
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
        df.dropna(subset=['ROMol'], inplace=True)
    print("Finished.")
    return df

def add_features(df, remove_na=False, pro=['MW', 'logP', 'HBA', 'HBD', 
                        'TPSA', 'NRB', 'MCF', 'SA', 'QED',
                        'rings', 'scaffold_inchi', 'scaffold_smiles']):
    if "ROMol" not in df.columns:
        df = validity_filter(df)
    df = parallelize_dataframe(df, add_descriptors)
    if remove_na:
        df.dropna(subset=pro, inplace=True)
    return df


def read_chembl_zip_file(file):
    df = pd.read_csv(file, compression='zip', sep=';')
    return autoFindsmile(df)

def add_standard_type(df, activity_col='IC50', type_='IC50', relation_col=None, 
                      c=None):
    if type_ == 'IC50':
        df['standard_type'] = 'IC50'
        if relation_col is None:
            df['standard_relation'] = '='
        df['standard_value'] = df[activity_col]
        del df[activity_col]
        if c is not None:
            df['standard_value'] = df['standard_value']  * c
        df['standard_units'] = 'nM'
    return df

def combine_mol_dfs(files, source, combine=True):
    """
    pay attentation to
        ID
        source
        bioactivity's column name should be IC50 or etc.
        unit should nM
        
    example:
    -----------------------
    files = ['/home/zdx/project/MDZT-1003/data/actives_org/SMS2_Takeda_Fudan.csv',
             '/home/zdx/project/MDZT-1003/data/actives_org/CHEMBL3112379.csv'
             ]
    
    dfs, df_o, df_d = combine_mol_dfs(files, [None, 'chembl'])
    """
    
    dfs = []
    dfs_small = []
    for file, s in zip(files, source):
        # file = files[1]
        file_format = fileFormat(file)
        if file_format == 'zip':
            df = read_chembl_zip_file(file)
        if file_format == 'csv':
            df = pd.read_csv(file)
        if not 'SMILES' in df.columns:
            df = autoFindsmile(df)
        if 'standard_type' not in df.columns:
            df = add_standard_type(df)
        if 'source' not in df.columns and s is not None:
            df['source'] = s
        if 'molecule_chembl_id' in df.columns:
            df.rename(columns={'molecule_chembl_id':'ID'}, inplace=True)
        if 'ID' not in df.columns:
            df['ID'] = addID(df, s)
        df = add_inchi(df)
        dfs.append(df)
        dfs_small.append(df[['ID',
                            'SMILES', 
                            'standard_value',
                            'standard_type',
                            'standard_units',
                            'standard_relation',
                            'source',
                            'InChI',
                             ]])
    df_small = pd.concat(dfs_small)
    df_small = add_features(df_small)
    del df_small['ROMol']
    df_small.sort_values(['standard_type', 'standard_value'], inplace=True)
    return dfs, df_small, df_small.drop_duplicates('InChI')

files = ['/home/zdx/project/MDZT-1003/data/actives_org/SMS2_Takeda_Fudan.csv',
         '/home/zdx/project/MDZT-1003/data/actives_org/CHEMBL3112379.csv'
         ]

dfs, df_o, df_d = combine_mol_dfs(files, [None, 'chembl'])

df_d.to_csv('/home/zdx/project/MDZT-1003/data/actives_org/SMS2_actives.csv', index=False)
