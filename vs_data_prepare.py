#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:23:14 2022

@author: zdx
"""

import os
import re
import numpy as np
import pandas as pd
from utils import build_new_folder, refinePDB, try_copy, GetFileName,\
    add_decoy_properties, saveVariable, batchPDB2fasta,MultiProteinAlignments,\
    hcluster,cluster2Fold, file2mol, mol2file, try_copy, ReadDataFrame
from tqdm import tqdm
from glob import glob
from complex_knn_feature import feature_extractor

out_dir = '/home/zdx/data/VS_dataset'
target_folder_name = '1_targets'
ref_lig_folder_name = '2_reference_ligands'

# """
# DUD-E
# """
# dataset_name = 'DUD-E'
# orginal_dir = None
# sub_out_dir = os.path.join(out_dir, dataset_name)

# if orginal_dir is None:
#     dude_url = 'http://dude.docking.org/db/subsets/all/all.tar.gz'
#     os.system(f'cd {out_dir} && wget {dude_url}')
#     os.system(f'cd {out_dir} && tar -xvzf all.tar.gz')
#     os.system(f'cd {out_dir} && mv all 0_raw')


# ## Create output root folder
# if not os.path.exists(sub_out_dir):
#     os.makedirs(sub_out_dir)

# ## orginial folder new place
# org_new_folder = os.path.join(sub_out_dir, '0_raw')

# targets = os.listdir(org_new_folder)

# ## Move orginal folder to output root folder
# # os.system(f'mv {orginal_dir} {org_new_folder}')

# ## move target pdb files 
# new_folder = os.path.join(sub_out_dir, target_folder_name)
# build_new_folder(new_folder)

# ## fgfr1 need to manual download from http://dude.docking.org/targets/fgfr1
# fgfr1_url = 'http://dude.docking.org/targets/fgfr1/fgfr1.tar.gz'

# os.system(f'cd {sub_out_dir} && wget {fgfr1_url}')

# fgfr1_path = os.path.join(sub_out_dir, 'fgfr1.tar.gz')
# os.system(f'cd {sub_out_dir} && tar -xvzf {fgfr1_path}')
# new_fgfr1_folder = os.path.join(sub_out_dir, 'fgfr1')

# fgfr1_org_folder = os.path.join(org_new_folder, 'fgfr1')
# os.system(f'rm -rf {fgfr1_org_folder}')
# os.system(f'mv {new_fgfr1_folder} {org_new_folder}')
# os.remove(fgfr1_path)

# for target in tqdm(targets):
#     sub_target_folder = os.path.join(org_new_folder, target)
#     target_pdb_file = os.path.join(sub_target_folder, 'receptor.pdb')
#     if os.path.exists(os.path.join(sub_target_folder, f'{target}.pdb')):
#         os.remove(os.path.join(sub_target_folder, f'{target}.pdb'))
#     process_target_pdb_file = os.path.join(new_folder, f'{target}.pdb')
#     refinePDB(target_pdb_file, process_target_pdb_file)

# ## Move reference ligand .mol2 files 
# new_folder = os.path.join(sub_out_dir, ref_lig_folder_name)
# build_new_folder(new_folder)

# for target in tqdm(targets):
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'crystal_ligand.mol2')
#     des = os.path.join(new_folder, f'{target}.mol2')
#     try_copy(source, des)
    
# ## Move actives mol2.gz
# new_folder = os.path.join(sub_out_dir, 'original_actives')
# build_new_folder(new_folder)

# for target in tqdm(targets):
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'actives_final.mol2.gz')
#     des = os.path.join(new_folder, f'{target}.mol2.gz')
#     try_copy(source, des)

# ## Move decoys mol2.gz
# new_folder = os.path.join(sub_out_dir, 'original_decoys')
# build_new_folder(new_folder)

# failed = []
# for target in tqdm(targets):
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'decoys_final.mol2.gz')
#     des = os.path.join(new_folder, f'{target}.mol2.gz')
#     try:
#         try_copy(source, des)
#     except:
#         failed.append(target)

# ## Move actives_final.ism files
# new_folder = os.path.join(sub_out_dir, 'original_actives')
# build_new_folder(new_folder)

# failed = []
# for target in tqdm(targets):
#     """
#     target = 'ampc'
#     """
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'actives_final.ism')
#     des = os.path.join(new_folder, f'{target}.csv')
#     df = pd.read_csv(source, sep=' ', header=None)
    
#     try:
#         if len(df.columns) == 3:
#             df.columns = ['SMILES', 'un_known', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#         elif len(df.columns) == 2:
#             df.columns = ['SMILES', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#     except:
#         failed.append(target)
        
# ## Move actives_final.ism files
# new_folder = os.path.join(sub_out_dir, 'original_actives')
# build_new_folder(new_folder)

# failed = []
# for target in tqdm(targets):
#     """
#     target ampc
#     """
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'actives_final.ism')
#     des = os.path.join(new_folder, f'{target}.csv')
#     df = pd.read_csv(source, sep=' ', header=None)

#     try:
#         if len(df.columns) == 3:
#             df.columns = ['SMILES', 'un_known', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#         elif len(df.columns) == 2:
#             df.columns = ['SMILES', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#     except:
#         failed.append(target)
        
# ## Move decoys_final.ism files
# new_folder = os.path.join(sub_out_dir, 'original_decoys')
# build_new_folder(new_folder)

# failed = []
# for target in tqdm(targets):
#     """
#     """
#     sub_target_folder = os.path.join(org_new_folder, target)
#     source = os.path.join(sub_target_folder, 'decoys_final.ism')
#     des = os.path.join(new_folder, f'{target}.csv')
#     df = pd.read_csv(source, sep=' ', header=None)
    
#     try:
#         if len(df.columns) == 3:
#             df.columns = ['SMILES', 'un_known', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#         elif len(df.columns) == 2:
#             df.columns = ['SMILES', 'ID']
#             df[['ID', 'SMILES']].to_csv(des, index=False)
#     except:
#         failed.append(target)

# ## Move deepcoy decoy

# deepcoy_url = 'http://opig.stats.ox.ac.uk/data/downloads/DeepCoy_decoys.tar.gz'
# os.system(f'cd {out_dir} && wget {deepcoy_url}')
# os.system(f'cd {out_dir} && tar -xvzf DeepCoy_decoys.tar.gz')

# new_folder = os.path.join(sub_out_dir, '5_deepcoy_decoys')
# build_new_folder(new_folder)

# files = glob(os.path.join(out_dir, 'DeepCoy_decoys', 'DeepCoy-DUDE-SMILES', '*.txt'))

# for file in tqdm(files):
#     # file = '/home/zdx/data/VS_dataset/DeepCoy_decoys/DeepCoy-DUDE-SMILES/dude-target-aces-decoys-final.txt'
#     df = pd.read_csv(file, sep=' ', header=None)
#     df.columns = ['active', 'SMILES']
#     actives = list(set(df.active))
#     target_name = GetFileName(file).split('-')[2]
    
#     dfs = []
#     for i, active in enumerate(actives):
#         # active = actives[0]
#         sub_df = df[df['active']==active]
#         sub_df['active_id'] = '_'.join([target_name, 'active', str(i)])
#         dfs.append(sub_df)
#     df = pd.concat(dfs)    
#     df['ID'] = ['_'.join([target_name, 'decoy', str(x)]) for x in range(0, len(df))]
#     des = os.path.join(new_folder, f'{target_name}.csv')
#     df.to_csv(des, index=False)
    
# ## Extract deepcoy active
# deepcoy_decoy_files = glob(os.path.join(new_folder, '*.csv'))
# new_folder = os.path.join(sub_out_dir, '6_deepcoy_actives')
# build_new_folder(new_folder)

# for file in tqdm(deepcoy_decoy_files):
#     # file = deepcoy_decoy_files[0]
#     target_name = GetFileName(file)
#     df = pd.read_csv(file)
#     df.drop_duplicates('active_id', inplace=True)
#     df.drop(columns=['SMILES', 'ID'], inplace=True)
#     df.columns = ['SMILES', 'ID']
#     des = os.path.join(new_folder, f'{target_name}.csv')
#     df.to_csv(des, index=False)
    
# """
# -------------------------------------------------------------------------------
# MUV ligand-based dataset
# """
# muv_url = 'https://static-content.springer.com/esm/art%3A10.1186%2F1758-2946-5-26/MediaObjects/13321_2013_467_MOESM5_ESM.gz'
# os.system(f'cd {out_dir} && wget {muv_url}')
# # gunzip -r MUV/
# # mv MUV/ 0_raw/

# dataset_name = 'MUV'
# sub_out_dir = os.path.join(out_dir, dataset_name)
# build_new_folder(sub_out_dir)
# muv_ori_dir = os.path.join(out_dir, '13321_2013_467_MOESM5_ESM', 'MUV')
# muv_new_dir = os.path.join(sub_out_dir, '0_raw')
# os.system(f'mv {muv_ori_dir} {muv_new_dir}')

# files = glob(os.path.join(muv_new_dir, '*'))

# new_folder = os.path.join(sub_out_dir, 'original_actives')
# build_new_folder(new_folder)

# new_folder = os.path.join(sub_out_dir, 'original_decoys')
# build_new_folder(new_folder)

# for file in files:
#     # file = files[0]
#     if 'active' in file:
#         df = pd.read_csv(file, sep='\t')
#         df.rename(columns={'# PUBCHEM_COMPOUND_CID': 'cid'}, inplace=True)
        
#         target_name = GetFileName(file).split('_')[3]
#         new_folder = os.path.join(sub_out_dir, 'original_actives')
#         des = os.path.join(new_folder, f'{target_name}.csv')
#         df.to_csv(des, index=False)
#     if 'decoy' in file:
#         df = pd.read_csv(file, sep='\t')
#         df.rename(columns={'# PUBCHEM_COMPOUND_CID': 'cid'}, inplace=True)
        
#         target_name = GetFileName(file).split('_')[3]
#         new_folder = os.path.join(sub_out_dir, 'original_decoys')
#         des = os.path.join(new_folder, f'{target_name}.csv')
#         df.to_csv(des, index=False)
        
"""
compute 26 feature
"""
# dataset_name = 'DUD-E'
# sub_out_dir = os.path.join(out_dir, dataset_name)

# files = []

# files += glob(os.path.join(sub_out_dir, 'original_actives', '*.csv'))
# files += glob(os.path.join(sub_out_dir, 'original_decoys', '*.csv'))
# files += glob(os.path.join(sub_out_dir, 'deepcoy_decoys', '*.csv'))
# files += glob(os.path.join(sub_out_dir, 'deepcoy_actives', '*.csv'))

# for file in tqdm(files):
#     # file = files[0]
#     df = pd.read_csv(file)
#     df = add_decoy_properties(df)
#     if 'active' in file:
#         df['label'] = 1
#     if 'decoy' in file:
#         df['label'] = 0
#     df.to_csv(file, index=False)
    
# dataset_name = 'MUV'
# sub_out_dir = os.path.join(out_dir, dataset_name)
# files = []

# files += glob(os.path.join(sub_out_dir, 'original_actives', '*.csv'))
# files += glob(os.path.join(sub_out_dir, 'original_decoys', '*.csv'))


# for file in tqdm(files):
#     # file = files[0]
#     df = pd.read_csv(file)
#     df = add_decoy_properties(df)
#     if 'active' in file:
#         df['label'] = 1
#     if 'decoy' in file:
#         df['label'] = 0
#     df.to_csv(file, index=False)
    
# """
# Multiple protein sequences alignment

# And use MSA data to cluster, then use cluster result to split data
# """
# dataset_name = 'DUD-E'
# sub_out_dir = os.path.join(out_dir, dataset_name)
# pdb_files = glob(os.path.join(sub_out_dir, f'{target_folder_name}/*.pdb'))
# # fasta_file = os.path.join(sub_out_dir, 'DUD-E.fst')
# # seqs, names = batchPDB2fasta(pdb_files, outfile=fasta_file)
# identities_file = os.path.join(sub_out_dir, 'identities.csv')
# ytdist, labels = MultiProteinAlignments(identities_file=identities_file)
# cluster_res = hcluster(ytdist, labels)
# fold_res_path = os.path.join(sub_out_dir, '3_fold.json')
# fold_res = cluster2Fold(cluster_res, outfile=fold_res_path)
# """
# -------------------------------------------------------------------------------
# DUD-E docking result
# """
# dataset_name = 'DUD-E'
# decoy_source = 'original'
# docktool = 'vinardo'
# raw_data_folder = os.path.join(out_dir, 'Koes')
# input_raw_folder_name = 'dude_vs' # docked_dude
# # url = 'http://bits.csb.pitt.edu/files/docked_dude.tar'
# # os.system(f'mkdir {raw_data_folder} && cd {raw_data_folder} && wget {url}')
# # tar –xvf docked_dude.tar
# input_folder = os.path.join(raw_data_folder, input_raw_folder_name)

# sub_active_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_actives_docked')
# sub_decoy_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_decoys_docked')
# if not os.path.exists(sub_active_out_dir):
#     os.makedirs(sub_active_out_dir)
# if not os.path.exists(sub_decoy_out_dir):
#     os.makedirs(sub_decoy_out_dir)

# targets = os.listdir(input_folder)
# for target in tqdm(targets[86:]):
#     # target = 'fgfr1'
#     tmp_active_sdf_file = os.path.join(sub_active_out_dir, f'{target}.{docktool}.sdf')
#     tmp_active_pdbqt_file = os.path.join(sub_active_out_dir, f'{target}.{docktool}.pdbqt')
#     tmp_decoy_sdf_file = os.path.join(sub_decoy_out_dir, f'{target}.{docktool}.sdf')
#     tmp_decoy_pdbqt_file = os.path.join(sub_decoy_out_dir, f'{target}.{docktool}.pdbqt')

#     input_target_folder = os.path.join(input_folder, target)
#     sdf_files = glob(os.path.join(input_target_folder, '*.sdf.gz'))
#     active_sdf_files = []
#     decoy_sdf_files = []
#     for file in sdf_files:
#         if 'actives' in file:
#             active_sdf_files.append(file)
#         elif 'decoys' in file:
#             decoy_sdf_files.append(file)
#     active_mols = file2mol(active_sdf_files)
    
#     for mol in active_mols:
#         mol.calccharges()
    
#     mol2file(active_mols, tmp_active_sdf_file)
#     mol2file(active_mols, tmp_active_pdbqt_file, split=True, split_mode=True)
    
#     decoy_mols = file2mol(decoy_sdf_files)
#     for mol in decoy_mols:
#         mol.calccharges()
#     mol2file(decoy_mols, tmp_decoy_sdf_file)
    
#     mol2file(decoy_mols, tmp_decoy_pdbqt_file, split=True, split_mode=True)

# # Extract Knn Feature
# dataset_name = 'DUD-E'
# decoy_source = 'original'
# complex_feature_type = 'knn'
# docktool = 'vinardo'
# input_raw_folder_name = 'dude_vs' # docked_dude

# raw_data_folder = os.path.join(out_dir, 'Koes')

# input_folder = os.path.join(raw_data_folder, input_raw_folder_name)

# sub_active_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_actives_docked')
# sub_decoy_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_decoys_docked')

# sub_active_feature_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_actives_docked_feature')
# sub_decoy_feature_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_decoys_docked_feature')
# if not os.path.exists(sub_active_feature_out_dir):
#     os.makedirs(sub_active_feature_out_dir)
# if not os.path.exists(sub_decoy_feature_out_dir):
#     os.makedirs(sub_decoy_feature_out_dir)

# failed = []
# targets = os.listdir(input_folder)
# for x in ['active', 'decoy']:
#     for target in tqdm(targets):
#         # 
#         try:
#             in_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_{x}s_docked')
#             in_dir = os.path.join(in_dir, f'{target}.{docktool}')
#             files = glob(os.path.join(in_dir, '*.pdbqt'))
#             out_file = os.path.join(out_dir, dataset_name, f'{decoy_source}_{x}s_docked_feature', f'{target}.{docktool}.{complex_feature_type}.pickle')
#             if os.path.exists(out_file):
#                 continue
#             feature_extractor(
#                 protein_file = os.path.join(out_dir, dataset_name, target_folder_name, f'{target}.pdb'),
#                 ligand_files = files,
#                 multi_pose = False,
#                 labeled = True,
#                 pickle_filename = out_file)
#         except:
#             failed.append([target, x])
# print(failed)

"""
-------------------------------------------------------------------------------
LIT-PCBA Docked data
"""
# url = 'http://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz'
# # os.system(f'cd {out_dir} && wget {url}')

# url = 'http://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz'
# # os.system(f'cd {out_dir} && wget {url}')
# # tar -zxvf 

# # David R. Koes
# url = "http://bits.csb.pitt.edu/files/gninavs/lit-pcba_vs.tgz"

# decoy_source = 'original'
# dataset_name = 'LIT-PCBA'

# input_folder = os.path.join(out_dir, 'Koes', 'lit-pcba')
# # Get all target ID
# targets = os.listdir(input_folder)
# # Create LIT-PCBA dir
# output_dir = os.path.join(out_dir, dataset_name)
# build_new_folder(output_dir)
# # Create LIT-PCBA target dir
# output_target_folder = os.path.join(output_dir, target_folder_name)
# build_new_folder(output_target_folder)
# # Create LIT-PCBA ref ligand dir
# output_ref_lig_folder = os.path.join(output_dir, ref_lig_folder_name)
# build_new_folder(output_ref_lig_folder)
# # Create LIT-PCBA active compound docked dir
# output_active_docked_folder = os.path.join(output_dir, f'{decoy_source}_actives_docked')
# build_new_folder(output_active_docked_folder)
# # Create LIT-PCBA decoy compound docked dir
# output_decoy_docked_folder = os.path.join(output_dir, f'{decoy_source}_decoys_docked')
# build_new_folder(output_decoy_docked_folder)

# failed = []
# for target in tqdm(targets):
#     # target = targets[0]
#     input_target_dir = os.path.join(input_folder, target)
#     files = glob(os.path.join(input_target_dir, '*_protein_aligned.pdb'))
#     for file in files:
#         # file = target_files[0]
#         try:
#             name = GetFileName(file).split('_')[0]
#             new_name = '.'.join([target, name])
#             new_path = os.path.join(output_target_folder, f'{new_name}.pdb')
#             if not os.path.exists(new_path):
#                 try_copy(file, new_path)
#         except:
#             failed.append([target, 'target', new_path])
    
#     pdb_names = []
#     files = glob(os.path.join(input_target_dir, '*_ligand.pdb'))
#     for file in files:
#         # file = target_files[0]
#         try:
#             name = GetFileName(file).split('_')[0]
#             new_name = '.'.join([target, name])
#             new_path = os.path.join(output_ref_lig_folder, f'{new_name}.pdb')
#             if not os.path.exists(new_path):
#                 try_copy(file, new_path)
#             pdb_names.append(name)
#         except:
#             failed.append([target, 'ref_ligand' ,new_path])
        
#     pdb_names = set(pdb_names)
        
#     docked_files = glob(os.path.join(input_target_dir, f'AID*docked.sdf.gz'))
#     for name in list(pdb_names):
#         # name = list(pdb_names)[0]
#         try:
#             actives_files = []
#             decoy_files = []
#             for mol_type in ['active', 'inactive']:
#                 for file in docked_files:
#                     filename = GetFileName(file)
#                     if re.search(r"_{mol_type}+.+{name}+".format(mol_type=mol_type, name=name), filename):
#                         if mol_type == 'active':
#                             actives_files.append(file)
#                         elif mol_type == 'inactive':
#                             decoy_files.append(file)
#             new_name = '.'.join([target, name])
            
#             tmp_active_sdf_file = os.path.join(output_active_docked_folder, f'{new_name}.smina.sdf')
#             tmp_active_pdbqt_file = os.path.join(output_active_docked_folder, f'{new_name}.smina.pdbqt')
#             tmp_decoy_sdf_file = os.path.join(output_decoy_docked_folder, f'{new_name}.smina.sdf')
#             tmp_decoy_pdbqt_file = os.path.join(output_decoy_docked_folder, f'{new_name}.smina.pdbqt')
            
#             if os.path.exists(tmp_active_sdf_file) and os.path.exists(tmp_decoy_sdf_file)\
#                 and len(os.listdir(tmp_active_pdbqt_file.replace('.pdbqt', '')))>1 and\
#                     len(os.listdir(tmp_decoy_pdbqt_file.replace('.pdbqt', '')))>1:
#                 continue
            
#             active_mols = file2mol(actives_files)
#             for mol in active_mols:
#                 mol.title = 'SID'+ mol.title
#                 mol.calccharges()
    
#             mol2file(active_mols, tmp_active_sdf_file)
#             mol2file(active_mols, tmp_active_pdbqt_file, split=True, split_mode=True)
            
#             decoy_mols = file2mol(decoy_files)
#             for mol in decoy_mols:
#                 mol.title = 'SID'+ mol.title
#                 mol.calccharges()
    
#             mol2file(decoy_mols, tmp_decoy_sdf_file)
#             mol2file(decoy_mols, tmp_decoy_pdbqt_file, split=True, split_mode=True)
#         except:
#             failed.append([target, 'docked', 'name'])
# print(failed)
"""
-------------------------------------------------------------------------------
LIT-PCBA 2D data
"""
# decoy_source = 'original'
# dataset_name = 'LIT-PCBA'
# input_folder = os.path.join(out_dir, dataset_name, '0_raw', 'full')
# targets = os.listdir(input_folder)
# for x in ['actives', 'inactives']:
#     if x == 'inactives':
#         x1 = 'decoys'
#     else:
#         x1 = x
#     sub_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_{x1}')
#     build_new_folder(sub_out_dir)
#     for target in targets:
#         if '.tgz' in target:
#             continue
#         # target = targets[0]
#         target_folder = os.path.join(input_folder, target)
#         infile = os.path.join(target_folder, f'{x}.smi')
#         outfile = os.path.join(sub_out_dir, f'{target}.full.csv')
#         df = ReadDataFrame(infile)
#         df['ID'] = [''.join(['SID', str(x)]) for x in df['ID']]
#         df.to_csv(outfile, index=False)

"""
compute 26 feature
"""
# sub_out_dir = os.path.join(out_dir, dataset_name)

# files = []

# files += glob(os.path.join(sub_out_dir, f'{decoy_source}_actives', '*.csv'))
# files += glob(os.path.join(sub_out_dir, f'{decoy_source}_decoys', '*.csv'))

# for file in tqdm(files):
#     # file = files[0]
#     df = pd.read_csv(file)
#     df = add_decoy_properties(df)
#     if 'active' in file:
#         df['label'] = 1
#     if 'decoy' in file:
#         df['label'] = 0
#     df.to_csv(file, index=False)

"""
-------------------------------------------------------------------------------
LIT-PCBA Docked data KNN feature
"""
# dataset_name = 'LIT-PCBA'
# decoy_source = 'original'
# complex_feature_type = 'knn'
# docktool = 'smina'
# print("LIT-PCBA KNN feature extract.")
# def get_dataset_target_ids(data_root, dataset_name, 
#                            target_folder_name='1_targets'):
#     target_files = glob(os.path.join(data_root, dataset_name, 
#                                           target_folder_name, '*.pdb'))
#     target_ids = []
#     for file in target_files:
#         target_ids.append(GetFileName(file))
#     return target_ids

# sub_active_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_actives_docked')
# sub_decoy_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_decoys_docked')

# sub_active_feature_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_actives_docked_feature')
# sub_decoy_feature_out_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_decoys_docked_feature')
# if not os.path.exists(sub_active_feature_out_dir):
#     os.makedirs(sub_active_feature_out_dir)
# if not os.path.exists(sub_decoy_feature_out_dir):
#     os.makedirs(sub_decoy_feature_out_dir)

# failed = []
# targets = get_dataset_target_ids(out_dir, dataset_name)
# for x in ['active', 'decoy']:
#     for target in tqdm(targets):
#         try:
#             in_dir = os.path.join(out_dir, dataset_name, f'{decoy_source}_{x}s_docked')
#             in_dir = os.path.join(in_dir, f'{target}.{docktool}')
#             files = glob(os.path.join(in_dir, '*.pdbqt'))
#             out_file = os.path.join(out_dir, dataset_name, f'{decoy_source}_{x}s_docked_feature', f'{target}.{docktool}.{complex_feature_type}.pickle')
#             if os.path.exists(out_file):
#                 continue
#             feature_extractor(
#                 protein_file = os.path.join(out_dir, dataset_name, target_folder_name, f'{target}.pdb'),
#                 ligand_files = files,
#                 multi_pose = False,
#                 labeled = True,
#                 pickle_filename = out_file)
#         except:
#             failed.append([target, x])
# print(failed)

"""
-------------------------------------------------------------------------------
QUickCoy 1st generation prepare
"""

# files = glob(f'/home/zdx/data/VS_dataset/DUD-E/original_actives/*.csv')
# dude_active = concat_dfs(files)

# initial generation
# files = glob('/home/zdx/project/decoy_generation/result/REAL_212/init_decoy/DUD-E/*.csv')
# df = concat_dfs(files)
# df_count = df.groupby(by='active_id').count()
# df_count.reset_index(inplace=True)
# df_count = df_count[['active_id', 'active']]
# df_count.columns = ['ID', 'decoy_count']
# df_count.drop_duplicates(subset='ID', inplace=True)
# dude_active.drop_duplicates(inplace=True)
# dude_active_quickcoy = dude_active.merge(df_count, how='left', on='ID')
# dude_active_quickcoy = adjust_order(dude_active_quickcoy, pro=['ID', 'SMILES', 'decoy_count'])
# dude_active_quickcoy.sort_values('decoy_count', inplace=True)
# dude_active_quickcoy.to_csv('/home/zdx/project/decoy_generation/result/REAL_212/decoy_number.csv', index=False)
# filtered decoy
# files = glob('/home/zdx/data/VS_dataset/DUD-E/quickcoy_decoys/*.csv')
# df = concat_dfs(files)
# df_count = df.groupby(by='active_id').count()
# df_count.reset_index(inplace=True)
# df_count = df_count[['active_id', 'active']]
# df_count.columns = ['ID', 'decoy_count']
# df_count.drop_duplicates(subset='ID', inplace=True)
# dude_active_quickcoy = dude_active.merge(df_count, how='left', on='ID')
# dude_active_quickcoy = adjust_order(dude_active_quickcoy, pro=['ID', 'SMILES', 'decoy_count'])
# dude_active_quickcoy.sort_values('decoy_count', inplace=True)
# dude_active_quickcoy.to_csv('/home/zdx/data/VS_dataset/DUD-E/QuickCoy_filtered_decoy_number.csv', index=False)