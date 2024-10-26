from rdkit import Chem
import numpy as np
from io import StringIO
import pickle
import time

smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5]


class MolGraph:
    def __init__(self):
        self.nodes = {}

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        for ntype in subgraph.nodes:
            self.nodes.setdefault(ntype, []).extend(subgraph.nodes[ntype])

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i: [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor] for neighbor in self_node.get_neighbors(neighbor_ntype)] for self_node in self.nodes[self_ntype]]


class Node:
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']

    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph


def array_rep_from_smiles(molgraph):
    arrayrep = {
        'atom_features': molgraph.feature_array('atom'),
        'bond_features': molgraph.feature_array('bond'),
        'atom_list': molgraph.neighbor_list('molecule', 'atom'),
        'rdkit_ix': molgraph.rdkit_ix_array()
    }

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    return arrayrep


def gen_descriptor_data(smiles_list):
    smiles_to_fingerprint_array = {}

    for i, smiles in enumerate(smiles_list):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            molgraph = graph_from_smiles(smiles)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph)
            smiles_to_fingerprint_array[smiles] = arrayrep
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            time.sleep(1)

    return smiles_to_fingerprint_array


def get_smiles_dicts(smiles_list):
    smiles_to_fingerprint_features = gen_descriptor_data(smiles_list)
    feature_dicts = {}

    max_atom_len = 0
    max_bond_len = 0

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_len = len(arrayrep['atom_features'])
        bond_len = len(arrayrep['bond_features'])
        max_atom_len = max(max_atom_len, atom_len)
        max_bond_len = max(max_bond_len, bond_len)

    # Initialize arrays for zero-padding
    max_atom_len += 1
    max_bond_len += 1
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}
    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}
    smiles_to_atom_mask = {}

    degrees = [0, 1, 2, 3, 4, 5]

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        # Zero-padded arrays
        atoms = np.zeros((max_atom_len, atom_features.shape[1]))
        bonds = np.zeros((max_bond_len, bond_features.shape[1]))
        atom_neighbors = np.full((max_atom_len, len(degrees)), max_atom_index_num, dtype=int)
        bond_neighbors = np.full((max_bond_len, len(degrees)), max_bond_index_num, dtype=int)
        mask = np.zeros(max_atom_len)

        # Fill arrays with real data
        atoms[:len(atom_features)] = atom_features
        bonds[:len(bond_features)] = bond_features
        mask[:len(atom_features)] = 1

        atom_neighbor_count = 0
        bond_neighbor_count = 0

        for degree in degrees:
            atom_neighbors_list = arrayrep.get(('atom_neighbors', degree), [])
            bond_neighbors_list = arrayrep.get(('bond_neighbors', degree), [])

            atom_neighbors[atom_neighbor_count:atom_neighbor_count + len(atom_neighbors_list), :] = atom_neighbors_list
            bond_neighbors[bond_neighbor_count:bond_neighbor_count + len(bond_neighbors_list), :] = bond_neighbors_list

            atom_neighbor_count += len(atom_neighbors_list)
            bond_neighbor_count += len(bond_neighbors_list)

        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds
        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        smiles_to_atom_mask[smiles] = mask

    feature_dicts.update({
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors
    })

    return feature_dicts


def one_of_k_encoding(x, allowable_set):
    """Return one-hot encoding of x in allowable_set."""
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=True):
    """Generate atom feature vector."""
    allowable_symbols = [
        'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
        'Se', 'Br', 'Te', 'I', 'At', 'other'
    ]
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    
    atom_symbol = one_of_k_encoding_unk(atom.GetSymbol(), allowable_symbols)
    atom_degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    formal_charge = [atom.GetFormalCharge()]
    radical_electrons = [atom.GetNumRadicalElectrons()]
    atom_hybridization = one_of_k_encoding_unk(atom.GetHybridization(), hybridization_types)
    is_aromatic = [atom.GetIsAromatic()]
    
    # Include hydrogen information if not explicitly included in input
    if not explicit_H:
        num_hydrogens = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    else:
        num_hydrogens = []

    chirality_features = []
    if use_chirality:
        chirality_features = [False, False, atom.HasProp('_ChiralityPossible')]
        if atom.HasProp('_CIPCode'):
            cip_code = atom.GetProp('_CIPCode')
            chirality_features = one_of_k_encoding_unk(cip_code, ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]

    # Return combined atom feature vector
    return np.array(atom_symbol + atom_degree + formal_charge + radical_electrons + 
                    atom_hybridization + is_aromatic + num_hydrogens + chirality_features)

def bond_features(bond, use_chirality=True):
    """Generate bond feature vector."""
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

    if use_chirality:
        bond_feats += one_of_k_encoding_unk(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    
    return np.array(bond_feats)

def num_atom_features():
    """Return the number of atom features."""
    simple_molecule = Chem.MolFromSmiles('CC')
    return len(atom_features(simple_molecule.GetAtoms()[0]))

def num_bond_features():
    """Return the number of bond features."""
    simple_molecule = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_molecule)
    return len(bond_features(simple_molecule.GetBonds()[0]))


def save_smiles_dicts(smilesList, filename):
    # Get fingerprint features once to avoid repetitive computation
    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    # Initialize max values and structure for storing results
    max_atom_len = max_bond_len = 0
    num_atom_features = num_bond_features = 0
    smiles_to_rdkit_list = {}
    
    # Loop to calculate the max atom and bond lengths, and store rdkit_ix list
    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len, num_atom_features = atom_features.shape
        bond_len, num_bond_features = bond_features.shape

        max_atom_len = max(max_atom_len, atom_len)
        max_bond_len = max(max_bond_len, bond_len)

    # Add 1 to max lengths for zero padding and initialize the container dictionaries
    max_atom_len += 1
    max_bond_len += 1
    max_atom_index_num = max_atom_len - 1
    max_bond_index_num = max_bond_len - 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}
    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}
    smiles_to_atom_mask = {}

    degrees = [0, 1, 2, 3, 4, 5]

    # Pre-compute arrays for neighbors, atom info, bond info, and masks
    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        mask = np.zeros(max_atom_len)
        atoms = np.zeros((max_atom_len, num_atom_features))
        bonds = np.zeros((max_bond_len, num_bond_features))
        atom_neighbors = np.full((max_atom_len, len(degrees)), max_atom_index_num)
        bond_neighbors = np.full((max_atom_len, len(degrees)), max_bond_index_num)

        mask[:len(atom_features)] = 1.0
        atoms[:len(atom_features)] = atom_features
        bonds[:len(bond_features)] = bond_features

        # Fill neighbor arrays based on the pre-computed data
        for degree in degrees:
            atom_neighbors_list = arrayrep.get(('atom_neighbors', degree), [])
            bond_neighbors_list = arrayrep.get(('bond_neighbors', degree), [])

            for i, degree_array in enumerate(atom_neighbors_list):
                atom_neighbors[i, :len(degree_array)] = degree_array

            for i, degree_array in enumerate(bond_neighbors_list):
                bond_neighbors[i, :len(degree_array)] = degree_array

        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds
        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        smiles_to_atom_mask[smiles] = mask

    # Dump all the generated data into a pickle file
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }

    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(feature_dicts, f)
    
    print(f'Feature dicts file saved as {filename}.pickle')
    return feature_dicts


def get_smiles_array(smilesList, feature_dicts):
    # Using list comprehensions for efficient data extraction
    x_mask = np.array([feature_dicts['smiles_to_atom_mask'][smiles] for smiles in smilesList])
    x_atom = np.array([feature_dicts['smiles_to_atom_info'][smiles] for smiles in smilesList])
    x_bonds = np.array([feature_dicts['smiles_to_bond_info'][smiles] for smiles in smilesList])
    x_atom_index = np.array([feature_dicts['smiles_to_atom_neighbors'][smiles] for smiles in smilesList])
    x_bond_index = np.array([feature_dicts['smiles_to_bond_neighbors'][smiles] for smiles in smilesList])

    return x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, feature_dicts['smiles_to_rdkit_list']


