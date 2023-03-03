#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue Feb 15 11:28:20 2022
# @author: zdx

import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from os_command_py import os_command

import glob
from argparse import ArgumentParser
from tqdm import tqdm

# from openbabel import pybel
from utils import GetFileFormat, GetFileName, csv2smi, smi2csv, ReadDataFrame,\
    SaveDataFrame, ispath, same_folder_file, mol2file, file2mol,\
    GetFileNameFormat, add_features, refinePDB, sdf2csv, ChaneFileSuffix,\
    pdbqts2sdf

def get_score_from_smina_log(file, name=None):
    """
    file = '/home/zdx/Downloads/aa2ar/test/smi_obabel/parallel/CHEMBL16687.smina.log'
    """
    if name is None:
        name = os.path.basename(file).split('.smina.log')[0]
    flag = 0
    score_lines = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if '-----+' in line and flag!=1:
                flag = 1
                continue
            if flag==1 and len(line)==40:
                score_lines.append(line.split())
            else:
                continue
    if len(score_lines) > 0:
        df = pd.DataFrame(score_lines)
        df.columns = ['ID', 'smina', 'lb_rmsd', 'ub_rmsd']
        df['smina'] = df['smina'].astype(float)
        df['lb_rmsd'] = df['lb_rmsd'].astype(float)
        df['ub_rmsd'] = df['ub_rmsd'].astype(float)
        df['ID'] = name + '_mode_' + df['ID']
        best_score = df['smina'][0]
        return df, best_score
    else:
        return np.nan, np.nan

# def mgltools_prepare_ligand(infile, outfile=None, informat=None, rigid=False, 
#                             outformat='pdbqt'):
#     """
#     mgltools_prepare_ligand('/home/zdx/Downloads/aa2ar/test/short_smi/CHEMBL130830.pdbqt')
    
#     infile = '/home/zdx/Downloads/aa2ar/test/smi/pybel_mgltools/pybel/CHEMBL1099028.pybel.pdbqt'
#     outfile = '/home/zdx/Downloads/aa2ar/test/smi/pybel_mgltools/prepared_compound/CHEMBL1099028.pdbqt'
#     informat=None
#     rigid=False
#     outformat='pdbqt'
    
#     """
#     if informat is None:
#         informat = GetFileFormat(infile)
#     if informat not in ['mol2', 'pdb', 'pdbqt']:
#         print('Input file only support mol2, pdb, pdbqt, please check.')
#         return
#     option = []
#     if rigid:
#         option.append('-Z')
#     if outfile is None:
#         outfile = infile.replace(f'.{informat}', f'.mgltools.{outformat}')
#     in_dir = os.path.dirname(infile)
#     command_text = ['cd', in_dir, '&&',
#                     './prepare_ligand4.py',
#                     "-l", os.path.basename(infile),
#                     "-B", 'none',
#                     "-o", outfile] + option
#     cmd = os_command.Command(command_text)
#     try:
#         cmd.run()
        
#     except:
#         print(f'Failed on {infile}.')
#         return False
#     return outfile
        
# def mgltools_prepare_ligand_main(infiles, out_dir=None, mode='loop', 
#                                  rigid=False):
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     if mode == 'loop':
#         failed_files = []
#         outfiles = []
#         for infile in tqdm(infiles):
#             # infile = '/home/zdx/Downloads/aa2ar/test/smi/pybel_mgltools/pybel/CHEMBL1099028.pybel.pdbqt'
#             if out_dir is None:
#                 outfile = None
#             else:
#                 outfile = os.path.join(out_dir, '%s.pdbqt'%GetFileName(infile))
#             if not mgltools_prepare_ligand(infile=infile, 
#                                     outfile=outfile,
#                                     rigid=rigid
#                                     ):
#                 failed_files.append(outfile)
#             else:
#                 outfiles.append(outfile)
    
#     print('%s files failed:'%str(len(failed_files)))
#     if len(failed_files) > 0:
#         for file in failed_files:
#             print(file)
#     return outfiles
    
def obabel_formater(infile, out_dir=None, outfilename='active', informat=None, 
                    outformat='pdbqt', id_key='ID',
                    addh=True, gen3d=True, addcharge=True,
                    id_index=-1, sep=' ', overwrite=True, ref_ligand=False):
    """
    out_dir=None
    informat=None
    outformat='pdbqt'
    addh=True
    gen3d=True
    addcharge=True
    id_index=-1
    sep=' '
    overwrite=True
    ref_ligand=True
    infile = ''
    """

    convert2sdf = True

    if informat is None:
        informat = GetFileFormat(infile)
        if informat == 'ism':
            informat = 'smi'

    if informat=='csv':
        infile = csv2smi(infile)
        informat = 'smi'
        
    if informat=='sdf':
        convert2sdf = False
        mols = file2mol(infile)
        mol = mols[0]
        if id_key in list(mol.data.keys()):
            names = []
            for mol in mols:
                names.append(mol.data[id_key])

    infilename = GetFileName(infile)
    
    if out_dir is None:
        if ref_ligand:
            out_dir = os.path.dirname(infile)
        else:
            out_dir = os.path.join(os.path.dirname(infile), f'{infilename}_obabel')

    if ref_ligand:
        out_file = os.path.join(out_dir, f'{infilename}.{outformat}')
        if os.path.exists(out_file):
            return out_file
    else:
        out_file = os.path.join(out_dir, f'{outfilename}.{outformat}')

    if ref_ligand:
        gen3d = False
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not ref_ligand:
        files = glob.glob(os.path.join(out_dir, '*.pdbqt'))
        if len(files) > 0:
            return files 

    cmd_text = ['obabel',
                '-i%s' % informat,
                infile,
                '-O', 
                out_file
                ]

    if not ref_ligand:
        cmd_text.append('-m')
    if addh:
        cmd_text.append('--AddPolarH')
    if gen3d:
        cmd_text.append('--gen3D')
    if addcharge:
        cmd_text.append('--partialcharge')

    cmd = os_command.Command(cmd_text)
    
    try:
        cmd.run()
    except:
        print(f'Failed to convert {infile} \nfrom {informat} to {outformat}.')
        return False
    
    if ref_ligand:
        return out_file
    """
    out_dir = '/home/zdx/project/MDZT-1003/SMS2/inhibitors/tmp'
    """
    
    files = glob.glob(os.path.join(out_dir, f'*.{outformat}'))
    outfiles = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        """
        file = '/home/zdx/project/MDZT-1003/SMS2/inhibitors/tmp/active1.pdbqt'
        """
        if informat == 'sdf':
            name = names[i]
        else:
            with open(file, 'r') as f:
                name = f.readline()[15:].strip().split()[-1]
        outfile = os.path.join(out_dir, f'{name}.pdbqt')
        outfiles.append(outfile)
        os.rename(file, outfile)
    if convert2sdf:
        if os.path.exists(ChaneFileSuffix(infile, 'sdf')):
            return
        pdbqts2sdf(outfiles, ChaneFileSuffix(infile, 'sdf'))
    return outfiles

# def pybel_formater(infile, library_name=None, informat=None, outformat='pdbqt', 
#                    save_out=True, out_dir=None, overwrite=True, 
#                    addh=True, gen3d=True, addcharge=True,
#                    id_index=-1, sep=' '):
#     """
#     This method is not stable !!!
    
#     pybel_formater('/home/zdx/Downloads/aa2ar/test/short_smi/actives_final.ism')
    
#     id_index:
#         if input file is smi, maybe it has more than two cols following SMILES,
#         choose one column as id.
#     sep:
#         Separator between smi file columns following SMILES
    
#     infile = '/home/zdx/Downloads/aa2ar/test/short_smi/actives_final.ism'
#     """
#     if informat is None:
#         informat = GetFileFormat(infile)
#         if informat == 'ism':
#             informat = 'smi'
#     if informat not in ['sdf', 'smi', 'csv', 'pdb', 'pdbqt']:
#         print('Input file only support sdf, smi, csv, pdb, pdbqt, please check.')
#         return
#     if library_name is None:
#          library_name = GetFileName(infile)
#     print(f'Input file format: {informat}')
#     mol_ids = []
#     out_files = []
#     for i, mol in enumerate(tqdm(pybel.readfile(informat, infile))):
#         # mol = list(pybel.readfile(informat, infile))
#         # extract molecule id
#         try:
#             if informat == 'smi':
#                 if ' ' in mol.title:
#                     mol.title = mol.title.split(sep)[id_index]
#             mol_id = mol.title
#         except:
#             mol_id = ''
#         if mol_id == '':
#             mol_id = '_'.join([library_name, str(i)])
#         mol_ids.append(mol_id)
#         if addh:
#             mol.addh()
#         # generate 3D conformer
#         if gen3d:
#             mol.make3D()
#         if addcharge:
#             mol.calccharges()
#         if out_dir is None:
#             out_dir = os.path.join(os.path.dirname(infile), 
#                                    f'{library_name}_{outformat}_pybel')
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
        
#         out_file = os.path.join(out_dir, f'{mol_id}.{outformat}')
#         if save_out:
#             mol.write(outformat, out_file, overwrite)
#             out_files.append(out_file)
#     print(f'\n\nCovert ligand format from {informat} to {outformat} success!'
#           f'\nResults are saved in {out_dir}.')
#     return out_files

# def prepare_receptor(infile, save_out=True, out_file=None, out_dir=None,
#                      target_name=None, 
#                      addh=True,
#                      informat=None, overwrite=True):
#     """
#     infile = '/home/zdx/project/MDZT-1003/2022.04.08.MDZT-1003.AI designed SMS2 inhibitors/Protein structure/AF-Q8NHU3-SMS2.pdb'
#     """
#     if informat is None:
#         informat = GetFileFormat(infile)
        
#     if informat not in ['pdb', 'pdbqt']:
#         print('Input file only support pdb, pdbqt, please check.')
#         return
    
#     if target_name is None:
#          target_name = GetFileName(infile)
         
#     if out_file is None:
#         if out_dir is None:
#             out_dir = os.path.dirname(infile)
#         out_file = os.path.join(out_dir, '%s.pdbqt' % target_name)
        
#     in_dir = os.path.dirname(infile)
#     cmd = os_command.Command(['cd', in_dir, '&&',
#                               './prepare_receptor4.py',
#                               "-r", infile,
#                               "-A", 'checkhydrogens',
#                               "-o", out_file])
#     try:
#         cmd.run()
#     except:
#         print(f'Prepare receptor failed!\n Input file is {infile}.')
#         return False
#     print(f'Prepare receptor success!\nPrepared file was saved in {out_file}.')
#     return out_file

def docking_smina(compound_file, docking_pose_file, docking_log_file=None, 
                  docking_score_file=None,
                  receptor_file=False, ref_ligand_file=False, compound_id=None,
        cpus=8, autobox_add=8, exhaustiveness=16, smina_log=True, seed=0,
        add_smiles=False):
    """
    Don't change argument variable order!!!
    Dock on a target
    """
    if compound_id is None:
        compound_id = GetFileName(compound_file)
    
    if not receptor_file and not ref_ligand_file:
        print("No receptor file or reference ligand file.")
        return
    
    for file in [ref_ligand_file, compound_file]:
        if GetFileFormat(file) != 'pdbqt':
            print(f'Format of {file}, should be pdbqt.')
            return False
        
    out_dir = os.path.dirname(docking_pose_file)
        
    if docking_log_file is None:
        docking_log_file = os.path.join(out_dir, f'{compound_id}.smina.log')
    if docking_score_file is None:
        docking_score_file = os.path.join(out_dir, f'{compound_id}.smina.score')
        
    if add_smiles:
        mol = file2mol(compound_file)[0]
        smiles = mol.smiles
        
    options = []
    if smina_log:
        options += ["--log", docking_log_file]
    cmd_list = ["./software/smina.static", 
                             "--seed", str(seed),
                             "-q",
                             "-r", receptor_file, 
                             "-l", compound_file,
                             "--autobox_ligand", ref_ligand_file,
                             "--autobox_add", str(autobox_add),
                             "--exhaustiveness", str(exhaustiveness),
                             "--cpu", str(cpus),
                             "-o", docking_pose_file] + options
    cmd_text = ' '.join(cmd_list)
    cmd = os_command.Command(cmd_list)
    try:
        if not os.path.exists(docking_pose_file) or \
            os.path.getsize(docking_pose_file) <= 0:
            cmd.run()
        if docking_log_file is not None and os.path.exists(docking_log_file):
            df, score = get_score_from_smina_log(docking_log_file, compound_id)
            df.to_csv(docking_score_file, index=False)
            if add_smiles:
                return {'ID': compound_id, 'SMILES': smiles, 'smina': score, 
                        'pose_file': docking_pose_file}
            else:
                return {'ID': compound_id, 'smina': score, 
                        'pose_file': docking_pose_file}
    except:
        print(cmd_text)
        print("failed")
        if os.path.exists(docking_pose_file):
            os.remove(docking_pose_file)
        if os.path.exists(docking_log_file):
            os.remove(docking_log_file)
        if add_smiles:
            return {'ID': compound_id, 'SMILES': smiles, 'smina': np.nan, 
                    'pose_file': np.nan}
        else:
            return {'ID': compound_id, 'smina': np.nan, 
                    'pose_file': np.nan}

def add_smina_score(mol_df_or_file, score_df_or_file, out_file=None, 
    save_out=False, on='ID', how='left', sort=True, add=True):
    """
    mol_df_or_file = '/home/zdx/project/MDZT-1003/SMS2/inhibitors/1420.csv'
    score_df_or_file = '/home/zdx/project/MDZT-1003/SMS2/dock/pocket106/scores.csv'
    out_file=None
    save_out=True
    on='ID'
    how='left'
    sort=True
    add=True
    """

    score_df = ReadDataFrame(score_df_or_file)

    if mol_df_or_file is not None:
        mol_df = ReadDataFrame(mol_df_or_file)
        if 'SMILES' in mol_df.columns and 'SMILES' in score_df.columns:
            del score_df['SMILES']
        mol_df[on] = mol_df[on].apply(str)
        score_df[on] = score_df[on].apply(str)
        merged = mol_df.merge(score_df, on=on, how=how)
    else:
        merged = score_df

    if sort:
        merged.sort_values('smina', inplace=True)

    if out_file is None:
        if save_out:
            if ispath(mol_df_or_file):
                out_file = same_folder_file(mol_df_or_file, 'SminaScored')
            else:
                print("You need to give 'out_file=' ")
    if add and 'SMILES' in merged.columns:
        merged = add_features(merged)
    if 'pose_file' in merged.columns:
        SaveDataFrame(merged.drop(columns='pose_file'), out_file)
    else:
        SaveDataFrame(merged, out_file)
    return merged

"""
receptor_file = '/home/zdx/Downloads/aa2ar/test/receptor.pdbqt'
ref_ligand_file = '/home/zdx/Downloads/aa2ar/test/crystal_ligand.pdbqt'
compound_files = glob.glob('/home/zdx/Downloads/aa2ar/test/smi_obabel/parallel/obabel/*.pdbqt')
compound_files = compound_files[:10]
out_dir = '/home/zdx/Downloads/aa2ar/test/smi_obabel/parallel'
cpus=8
autobox_add=8
exhaustiveness=16
n_jobs=-1
smina_log=True
mode='parallel'
"""

def docking_smina_main(receptor_file, ref_ligand_file, compound_files, out_dir,
                       cpus=8, autobox_add=8, exhaustiveness=16, n_jobs=-1,
                       smina_log=True, mode='loop', org_file=None):
    
    if GetFileFormat(ref_ligand_file) != 'pdbqt':
        print(f'Format of {ref_ligand_file}, should be pdbqt.')
        return
        
    for compound_file in compound_files:
        if GetFileFormat(compound_file) != 'pdbqt':
            print(f'Format of {compound_file}, should be pdbqt.')
            return
    if isinstance(compound_files, str):
        compound_files = [compound_files]
    else:
        compound_files = list(compound_files)
        
    docking_pose_files = []
    if smina_log:
        docking_log_files = []
    compound_ids = []
    docking_socre_files = []
    
    for compound_file in compound_files:
        compound_id = GetFileName(compound_file)
        compound_ids.append(compound_id)
        docking_pose_file = os.path.join(out_dir, 'docking',
                                         f'{compound_id}.smina.pdbqt')
        docking_pose_files.append(docking_pose_file)
        if smina_log:
            docking_log_file = os.path.join(out_dir, 'docking',
                                            f'{compound_id}.smina.log')
            docking_log_files.append(docking_log_file)
        docking_score_file = os.path.join(out_dir, 'docking',
                                        f'{compound_id}.smina.score')
        docking_socre_files.append(docking_score_file)
                    
    if not os.path.exists(os.path.join(out_dir, 'docking')):
        os.makedirs(os.path.join(out_dir, 'docking'))
        print("Create docking folder.")

    print("Work mode:", mode)

    if mode == 'parallel':
        cpus = 1
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count() - 1
        pool = ThreadPool(n_jobs)
        print('CPUS:', n_jobs)
        scores = pool.starmap(partial(docking_smina,
                                receptor_file=receptor_file, 
                                ref_ligand_file=ref_ligand_file,
                                cpus=cpus, autobox_add=autobox_add, 
                                exhaustiveness=exhaustiveness, 
                                smina_log=smina_log), 
                                zip(compound_files,
                                    docking_pose_files,
                                    docking_log_files,
                                    docking_socre_files))
        pool.close()
        pool.join()
    elif mode == 'loop':
        scores = []
        print('CPUS:', cpus)
        for compound_file, docking_pose_file, docking_log_file,\
            docking_score_file in tqdm(zip(
            compound_files, docking_pose_files, docking_log_files,
            docking_socre_files), total=len(compound_files)):
                
            score = docking_smina(
                compound_file=compound_file,
                docking_pose_file=docking_pose_file,
                docking_log_file=docking_log_file,

                docking_score_file=docking_score_file,

                receptor_file=receptor_file, 
                ref_ligand_file=ref_ligand_file,
                
                cpus=cpus, 
                autobox_add=autobox_add, 
                exhaustiveness=exhaustiveness,
                smina_log=smina_log)
            scores.append(score)

    print("Finished docking.")
    df = pd.DataFrame(scores)
    out_file = os.path.join(out_dir, 'scores.csv')
    df = add_smina_score(org_file, df, out_file)
    sdf_file = os.path.join(out_dir, 'scores.sdf')
    GenerateSDFformDataFrame(df, sdf_file)
    return df

def SplitPDBQT(file, name=None, scores=None):
    """
    file = '/home/zdx/Documents/tmp/SMS2_00000.smina.pdbqt'
    mol = SplitPDBQT(file)
    """
    if not isinstance(file, str):
        return None

    out_dir = os.path.dirname(file)
    filename, fileformat = GetFileNameFormat(file)
    
    out_dir = os.path.join(out_dir, filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    flag = 0
    sub_mode = []
    count = 1
    best_mode_file = os.path.join(out_dir, f'mode_{count}.pdbqt')
    
    if os.path.exists(best_mode_file) and os.path.getsize(
        best_mode_file) > 0:
        try:
            best_mol = file2mol(best_mode_file)[0]
        except:
            best_mol = None
        return best_mol

    with open(file, 'r') as f:
        for line in f.readlines():
            if 'MODEL' in line or flag==1:
                if flag != 1:
                    flag = 1
                if 'ENDMDL' in line:
                    flag = 0
                sub_mode.append(line)
                if flag == 1:
                    continue
                if flag==0 and len(sub_mode)>0 and 'ENDMDL' in line:
                    sub_file = os.path.join(out_dir, f'mode_{count}.pdbqt')
                    with open(sub_file, 'w') as f:
                        for l in sub_mode:
                            if l[:4]=='ATOM':
                                f.write(l)
                    sub_mode = []                        
                    count += 1
    try:
        best_mol = file2mol(best_mode_file)[0]
    except:
        best_mol = None
    return best_mol

def GenerateSDFformDataFrame(df, outfile, mols=None, pdbqt_col='pose_file', 
    id_col='ID'):
    """
    df = merged
    mols=None
    pdbqt_col='pose_file'
    id_col='ID'
    outfile = '/home/zdx/project/MDZT-1003/SMS2/dock/pocket106/scores_2.sdf'
    """
    df.dropna(subset=[pdbqt_col], inplace=True)

    cols = list(df.columns)
    if pdbqt_col in cols:
        cols.remove(pdbqt_col)
    if pdbqt_col in cols:
        cols.remove(pdbqt_col)
    ids = list(df[id_col].values)
    
    if mols is None:
        if pdbqt_col in df.columns:
            files = list(df[pdbqt_col].values)
        else:
            print('Input dataframe do not include pose_file')
            return
        mols = []
        for i, (file, id_) in enumerate(zip(files, ids)):
            """
            file = files[0]
            id_ = ids[0]
            """
            mol = SplitPDBQT(file)
            if mol is None:
                continue
            mol.title = id_
            mol.data.clear()
            mol.data.update(df[cols].iloc[i,:].to_dict())
            mols.append(mol)
    else:
        for mol in mols:
            if mol is None:
                mols.remove(mol)
                continue
            mol.title = id_
            mol.data.clear()
            mol.data.update(df[cols].iloc[i,:].to_dict())
    mol2file(mols, outfile)

"""

"""
def docking_main(receptor_file, ref_ligand_file, out_dir,
                 ligand_lib=None, 
                 formater='obabel',
                 compound_files=None,
                 parallel=True, in_dir=None, cpus=16, autobox_add=8, n_jobs=-1,
                 smina_log=True,
                 exhaustiveness=16, docking_tool='smina'):

    ligand_lib_format = GetFileFormat(ligand_lib)
    if ligand_lib is not None:
        if ligand_lib_format == 'csv':
            org_file = ligand_lib
        elif ligand_lib_format == 'sdf':
            org_file = sdf2csv(ligand_lib)
        elif ligand_lib_format == 'smi':
            org_file = smi2csv(ligand_lib)
        else:
            org_file = None
    
    start = time.time()

    if parallel:
        mode = 'parallel'
    else:
        mode = 'loop'

    if receptor_file is None:
        print('Not given receptor file.')
        return
    if ref_ligand_file is None:
        print('Not given reference ligand file.')
        return
    
    print("Receptor input file:", receptor_file)
    print("Reference ligand input file:", ref_ligand_file)

    if in_dir is not None:
        compound_files = glob.glob(os.path.join(in_dir, '*.pdbqt'))
        print("Input folder includes", len(compound_files), 'pdbqt files.')

    if ligand_lib is not None:
        if formater == 'obabel':
            print("Use openbabel to process compound library.")
            compound_files = obabel_formater(ligand_lib)
            print("Finished processing compound library.")
        
    if GetFileFormat(ref_ligand_file)!='pdbqt':
        print("Use openbabel to process reference ligand.")
        ref_ligand_file = obabel_formater(ref_ligand_file, ref_ligand=True)
        print("Finished processing reference ligand.")
    
    if docking_tool == 'smina':
        print("\nDocking by Smina.")
        df = docking_smina_main(
            receptor_file = receptor_file,
            ref_ligand_file = ref_ligand_file,
            compound_files = compound_files,
            out_dir=out_dir,
            mode=mode,
            cpus=cpus, 
            autobox_add=autobox_add,
            exhaustiveness=exhaustiveness, 
            n_jobs=n_jobs,
            smina_log=smina_log,
            org_file=org_file
            )
    end = time.time()
    print("Time cost:", end-start)
    return df

def docking_target(in_dir, target, decoy_set):
    receptor = os.path.join(in_dir, '1_targets', f'{target}.pdb')
    ref_ligand = os.path.join(in_dir, '2_reference_ligands', 
                                    f'{target}.mol2')
    ligand_lib = os.path.join(in_dir, decoy_set, f'{target}.csv')

    tmp_out_dir = os.path.join(in_dir, decoy_set, f'{target}_smina')

    docking_main(
        receptor_file = receptor,
        ref_ligand_file=ref_ligand,
        ligand_lib = ligand_lib,
        out_dir = tmp_out_dir
                    )

def docking_dataset(in_dir, decoy_set):
    targets = list(pd.read_csv(os.path.join(in_dir, 'summary.csv'))['target_id'])

    failed = []
    for target in tqdm(targets):
        try:
            docking_target(in_dir, target, decoy_set)
        except:
            failed.append(target)
    print(failed)

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('-r', '--receptor', default=None, 
        help='A receptor file, pdb or pdbqt.')
    ap.add_argument('-f', '--ref_ligand', default=None,
        help='A reference ligand file, sdf, mol, mol2, pdb, pdbqt, etc.')
    ap.add_argument('-l', '--ligand_lib', default=None,
        help='A compound library file, csv, smi, sdf, pdb, pdbqt, etc.\n'
             'smi should include two columns, first column is SMILES, second '
             'is ID\n'
             'csv file should include SMILES column name and ID column name.'
             'Suggest csv, smi, sdf as input.\n'
             'Like:\n'
             '-l /home/user/demo.sdf'
        )
    ap.add_argument('-i', '--in_dir', default=None,
        help='If you alreay have a folder which includes a lot of compound files,'
             'and each compound file includes a single compound. use the parameter '
             'to give the folder path.\n'
             'Like:\n'
             '-i /home/user/target/')
    ap.add_argument('-c', '--compound_files', nargs='+', default=None,
        help='You can directly give a series of compound files for docking.\n'
             'Like: \n'
             '-c compound1.pdbqt compound2.pdbqt')
    ap.add_argument('-s', '--smiles', default=None, 
        help='You can directly give a ton of SMILES directly.\n'
             'Like:\n'
             '-s O=C(Nc1ccco1)c1ccccc1C(=O)N1CCCCC1 O=C(Nc1cccnn1)c1cccnc1')
    ap.add_argument('-o', '--out_dir', required=True, 
        help='Output folder to save the docking results.')
    ap.add_argument('--formater', default='obabel', choices=['obabel'],
        help='Tool is used to preprocess input file, default is obabel.')
    ap.add_argument('-p', '--parallel', default=False, action='store_true',
        help='Wether use parallel computing capability, default is False.')
    ap.add_argument('--docking_tool', default='smina', choices=['smina'], 
        help='Docking tool, default is Smina.')
    ap.add_argument('--only_prepare', action='store_true', default=False,
        help='Only preprocess without other operations,default is False.')
    ap.add_argument('--only_docking', action='store_true', default=False,
        help='Only docking without other operations,default is False.')
    
    # Smina parameters
    ap.add_argument('--cpus', default=8,
        help="This is a config of each Smina process's cpu.")
    ap.add_argument('--autobox_add', default=8,
        help='Amount of buffer space to add to auto-generated box'
             ' (default +8 on all six sides)')
    ap.add_argument('--exhaustiveness', default=16,
        help='exhaustiveness of the global search (roughly proportional to time)')
    ap.add_argument('--n_jobs', default=-1, 
        help='If you choose to use parallel computing capability, and this '
             'parameter is to set the number of cpus you want to use, default '
             'is -1, which means to use all the cpus your machine has.')
    ap.add_argument('--smina_log', default=True, action='store_false', 
        help='Whether save out log file of smina, default is True.')
    args = ap.parse_args()
    print("Arguments:")
    print(args)
    print()
    docking_main(
        receptor_file = args.receptor,
        ref_ligand_file = args.ref_ligand,
        compound_files = args.compound_files,
        formater = args.formater,
        out_dir = args.out_dir,
        ligand_lib = args.ligand_lib,
        in_dir = args.in_dir,
        cpus=args.cpus,
        autobox_add=args.autobox_add,
        exhaustiveness=args.exhaustiveness,
        n_jobs=args.n_jobs,
        smina_log=args.smina_log,
        parallel=args.parallel,
        docking_tool = args.docking_tool
        )

"""
./docking.py -r /home/zdx/project/MDZT-1003/2022.04.08.MDZT-1003.AI designed SMS2 inhibitors/Protein structure/AF-Q8NHU3-SMS2.pdb -f /home/zdx/Downloads/aa2ar/test/crystal_ligand.pdbqt -l /home/zdx/Downloads/aa2ar/test/smi_obabel/actives_final.ism -o /home/zdx/Downloads/aa2ar/test/smi_obabel/parallel
"""