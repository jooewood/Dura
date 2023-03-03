#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Aug 18 09:29:37 2022

# @author: zdx
# """

import glob
import os
import pickle
import numpy as np
from tqdm import tqdm
from utils import GetFileFormat, pdb2pdbqt, GetFileName

def get_type_dict(file_name):
    type_dict = {}
    with open(file_name) as fp:
        for i, line in enumerate(fp):
            type_dict[line.strip()] = i + 1
    return type_dict

def get_bin_list(lo, hi, step):
    return [ (lo + step * i) for i in \
        range(int((hi - lo) / step) + 1)] + [float('inf')]

def get_bin_id(num, bin_list):
    for i, v in enumerate(bin_list):
        if num < v:
            ans = i
            break          
    return ans     

ATOM_DICT = {'H': 1, 'HD': 2, 'HS': 3, 'C': 4, 'A': 5, 'N': 6, 'NA': 7, \
    'NS': 8, 'OA': 9, 'OS': 10, 'F': 11, 'Mg': 12, 'MG': 13, 'P': 14, \
    'SA': 15, 'S': 16, 'Cl': 17, 'CL': 18, 'Ca': 19, 'CA': 20, 'Mn': 21, \
    'MN': 22, 'Fe': 23, 'FE': 24, 'Zn': 25, 'ZN': 26, 'Br': 27, 'BR': 28, \
    'I': 29, 'Z': 30, 'G': 31, 'GA': 32, 'J': 33, 'Q': 34}  
AMINO_DICT = {'ZN5': 1, 'TRP': 2, 'GLZ': 3, 'GLU': 4, 'ASP': 5, 'ASN': 6, \
    'ALB': 7, 'GLN': 8, 'LEU': 9, 'WAU': 10, 'ASZ': 11, 'PHE': 12, 'MET': 13, \
        'HID': 14, 'HIS': 15, 'THR': 16, 'ARG': 17, 'SAM': 18, 'HIE': 19, \
            'WAM': 20, 'PRO': 21, 'SEM': 22, 'ILE': 23, 'TYR': 24, 'ASM': 25, \
                'ALA': 26, 'LYS': 27, 'SER': 28, 'CYS': 29, 'GLY': 30, \
                    'MG': 31, 'TYM': 32, 'VAL': 33}  
CHARGE_BIN = get_bin_list(-1, 1, 0.05)
DISTANCE_BIN = get_bin_list(0, 5.1, 0.3)


def atom(atom_str):
    '''
    Return information of an atom: list
    index:
        0 - amino
        1 - xyz
        2 - charge
        3 - atom_type
    '''
    atom_info = [AMINO_DICT.get(atom_str[17: 20].strip(), 0),
        [float(atom_str[30: 38]), float(atom_str[38: 46]), \
            float(atom_str[46: 54])],
        get_bin_id(float(atom_str[70: 76]), CHARGE_BIN),
        ATOM_DICT.get(atom_str[77: 79].strip(), 0)]
    return atom_info

class pose(object):
    def __init__(self, pose_name, pose_str_block):
        self.pose_name = pose_name
        self.atoms = [atom(atom_str) for atom_str in pose_str_block]
        self.atom_num = len(self.atoms)
        
class ligand(object):
    def __init__(self, ligand_file, multi_pose=False):
        self.ligand_name = GetFileName(ligand_file) 
        self.poses = []
        self.smiles = ''
        with open(ligand_file) as fp:
            block_flag = False
            pose_idx = 0            
            for line in fp.readlines():
                if line[0: 4] == 'ROOT' and block_flag is False:
                    block_flag = True
                    # pose_name = \
                    #     '{:s}_pose_{:0>2d}'.format(self.ligand_name, pose_idx)
                    pose_name = self.ligand_name
                    pose_str_block = []
                elif line[0: 4] == 'ATOM' and block_flag is True:
                    pose_str_block.append(line)
                elif line[0: 7] == 'ENDROOT':                    
                    current_pose = pose(pose_name, pose_str_block)
                    self.poses.append(current_pose)
                    self.atom_num = len(pose_str_block)
                    block_flag = False
                    pose_idx += 1
                    if not multi_pose:
                        break
                else:
                    continue
        self.pose_num = len(self.poses)
    
    def set_smiles(self, smiles_str):
        self.smiles = smiles_str

class protein(object):
    def __init__(self, protein_file):
        self.protein_name = GetFileName(protein_file)
        self.atoms = []
        with open(protein_file) as fp:
            for line in fp.readlines():
                if line[0: 4] == 'ATOM':
                    self.atoms.append(atom(line))
        self.atom_num = len(self.atoms)
        
def compute_atom_square_distance(coo1, coo2):
    return (coo1[0] - coo2[0]) ** 2 + (coo1[1] - coo2[1]) ** 2 + \
        (coo1[2] - coo2[2]) ** 2

def compute_square_distance_in_protein(atom_xyz_vec, protein_xyz_mat):
    return np.sum(np.square(protein_xyz_mat - atom_xyz_vec), axis=1)

def get_neighbors(molecule_pose, protein, kc=6, kp=2, max_atom_num=100):
    # print("into neighbors")
    inside_square_distance_matrix = np.zeros([molecule_pose.atom_num, \
        molecule_pose.atom_num])
    outside_square_distance_matrix = []
    for i in range(molecule_pose.atom_num):
        for j in range(0, i):
            inside_square_distance_matrix[i][j] = \
                inside_square_distance_matrix[j][i]
        for j in range(i, molecule_pose.atom_num):
            inside_square_distance_matrix[i][j] = \
                compute_atom_square_distance(molecule_pose.atoms[i][1], \
                    molecule_pose.atoms[j][1])   
        atom_xyz = np.array(molecule_pose.atoms[i][1])  
        protein_xyz_mat = np.array([protein.atoms[j][1] for j in \
            range(0, protein.atom_num)])
        outside_square_distance_matrix.append(\
            compute_square_distance_in_protein(atom_xyz, protein_xyz_mat))

    inside_neighbor_idx = \
        np.argsort(inside_square_distance_matrix, axis=1)[ : , : kc]
    outside_neighbor_idx = \
        np.argsort(outside_square_distance_matrix, axis=1)[ : , : kp]
    neighbor_atom = np.zeros([max_atom_num, kc + kp])
    neighbor_charge = np.zeros([max_atom_num, kc + kp])
    neighbor_distance = np.zeros([max_atom_num, kc + kp])
    neighbor_amino = np.zeros([max_atom_num, kp])
    mask = molecule_pose.atom_num
    for i in range(molecule_pose.atom_num):
        for j, neighbor_idx in enumerate(inside_neighbor_idx[i]):
            neighbor_atom[i][j] = molecule_pose.atoms[neighbor_idx][3]
            neighbor_charge[i][j] = molecule_pose.atoms[neighbor_idx][2]
            neighbor_distance[i][j] = get_bin_id(np.sqrt(\
                inside_square_distance_matrix[i][neighbor_idx]), DISTANCE_BIN)
        for j, neighbor_idx in enumerate(outside_neighbor_idx[i]):
            neighbor_atom[i][j + kc] = protein.atoms[neighbor_idx][3]
            neighbor_charge[i][j + kc] = protein.atoms[neighbor_idx][2]
            neighbor_distance[i][j + kc] = get_bin_id(np.sqrt(\
                outside_square_distance_matrix[i][neighbor_idx]), DISTANCE_BIN)
            neighbor_amino[i][j] = protein.atoms[neighbor_idx][0]
    # print("out neighbor")
    return [molecule_pose.pose_name, neighbor_atom, neighbor_charge, \
        neighbor_distance, neighbor_amino, mask]  

def feature_extractor(protein_file, ligand_files, multi_pose, labeled, 
                    pickle_filename):
    """
    protein_file = '/home/zdx/data/VS_dataset/DUD-E/1_targets/aa2ar.pdb'
    """
    
    print(f"Starting to extract {protein_file} feature ...")
    if GetFileFormat(protein_file) == 'pdb':
        protein_file = pdb2pdbqt(protein_file, protein=True)[0]
    
    ref_protein = protein(protein_file)
    
    name_list = []
    atom_matrix = []
    charge_matrix = []
    distance_matrix = []
    amino_matrix = []
    mask_vector = [] 
    y_matrix = []
    n = len(ligand_files)
    for i, ligand_file in enumerate(tqdm(ligand_files)):
        if os.stat(ligand_file).st_size == 0:
            print(ligand_file, "has no content.")
            continue
        if (i + 1) % 100 == 0:
            print('\r%d / %d...' %(i + 1, n), end='')
        lig = ligand(ligand_file, multi_pose=multi_pose)
        for lig_pose in lig.poses:
            try:                    
                if labeled is True:
                    if 'active' in ligand_file: 
                        y_matrix.append([0, 1])
                    elif 'decoy' in ligand_file:
                        y_matrix.append([1, 0]) 
                    else:
                        break
    
                lig_neighbor_info = get_neighbors(lig_pose, ref_protein)
                
                name_list.append(lig_neighbor_info[0])
                atom_matrix.append(lig_neighbor_info[1])
                charge_matrix.append(lig_neighbor_info[2])
                distance_matrix.append(lig_neighbor_info[3])
                amino_matrix.append(lig_neighbor_info[4])
                mask_vector.append(lig_neighbor_info[5])
            except:
                print(ligand_file)
                print(lig.ligand_name)
                
    atom_matrix = np.array(atom_matrix)
    charge_matrix = np.array(charge_matrix)
    distance_matrix = np.array(distance_matrix)
    amino_matrix = np.array(amino_matrix)
    mask_vector = np.array(mask_vector)
    y_matrix = np.array(y_matrix)

    target_id_list = [ref_protein.protein_name] * len(name_list)

    with open(pickle_filename, 'wb') as fp:
        print(pickle_filename)
        if labeled is True:
            pickle.dump([name_list, atom_matrix, charge_matrix, 
                distance_matrix, amino_matrix, mask_vector, target_id_list,
                y_matrix], fp, protocol=4)
        else:
            pickle.dump([name_list, atom_matrix, charge_matrix, 
                distance_matrix, amino_matrix, mask_vector, 
                target_id_list], fp, protocol=4)
                
    print('Name:', pickle_filename)
    print('Num:', len(name_list))
    if labeled is True:
        negative_num, positive_num = np.sum(y_matrix, axis=0)
        print('Positive Num:', positive_num)
        print('Negative Num:', negative_num)
    print(f"Extracted feature file was saved in {pickle_filename}.")
    return pickle_filename

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-p', '--protein_file', required=True, 
        help="Path of protein pdbqt file after docking.")
    ap.add_argument('-l', '--ligand_files', required=False, nargs='+',
        help="Paths of ligand pdbqt files after docking with the target.")
    ap.add_argument('-l_p', '--ligand_path', required=False,
        help="Path of folder of ligand pdbqt files after docking with the target.")
    ap.add_argument('-o', '--pickle_filename', required=True,
        help="Path of binary file which saved the extracted feature.")
    ap.add_argument('-m', '--multi_pose', action='store_true', default=False,
        help="Whether to extract multi binding pose feature.")
    ap.add_argument('--nolabel', action='store_false', default=True,
        help="If set this flag, the extractor will not add label.")
    args = ap.parse_args()
    if args.ligand_files is not None:
        ligand_files = args.ligand_files
    if args.ligand_path is not None:
        ligand_files = glob.glob(os.path.join(args.ligand_path, '*.pdbqt'))
    feature_extractor(protein_file = args.protein_file,
                      ligand_files = ligand_files,
                      multi_pose = args.multi_pose,
                      labeled = args.nolabel,
                      pickle_filename = args.pickle_filename
    )