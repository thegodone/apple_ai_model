import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

class CosineAnnealingWarmRestartsModified(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, factor=1.0, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.factor = factor
        self.eta_min = eta_min
        self.last_restart_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.last_restart_epoch:
            return [base_lr for base_lr in self.base_lrs]
        else:
            T_cur = self.last_epoch - self.last_restart_epoch
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / self.T_0)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check if the next epoch will trigger a restart, adjust learning rate ahead of that
        if self.last_epoch - self.last_restart_epoch + 1 == self.T_0:
            self.base_lrs = [base_lr * self.factor for base_lr in self.base_lrs]
            self.last_restart_epoch = self.last_epoch + 1  # schedule next restart after the upcoming epoch

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.get_lr()[i]


def gather_neighbors(features, degree_list):
    """
    Efficiently gathers neighbor features using the degree list.
    `features`: [batch_size, mol_length, num_features]
    `degree_list`: [batch_size, mol_length, max_neighbor_num]
    """
    batch_size, mol_length, num_features = features.size()

    # Ensure that degree_list has the right shape for expanding along the feature dimension
    degree_list_expanded = degree_list.unsqueeze(-1).expand(batch_size, mol_length, degree_list.size(-1), num_features)
    
    # Gather the neighbors along the correct dimension
    gathered_features = torch.gather(features.unsqueeze(2).expand(-1, -1, degree_list.size(-1), -1), 1, degree_list_expanded)
    
    return gathered_features
    
def gather_neighbors_bonds(bond_list, bond_degree_list):
    """
    Efficiently gathers bond neighbor features using the bond degree list.
    `bond_list`: [batch_size, num_bonds (63), num_bond_features]
    `bond_degree_list`: [batch_size, mol_length (56), max_neighbor_num (6)]
    """
    batch_size, num_bonds, num_bond_features = bond_list.size()
    mol_length, max_neighbor_num = bond_degree_list.size(1), bond_degree_list.size(2)
    
    # Step 1: Clamp bond_degree_list values to ensure they do not exceed the number of bonds
    bond_degree_list_clamped = torch.clamp(bond_degree_list, max=num_bonds-1)
    
    # Step 2: Expand bond_degree_list to match the bond features dimensions
    bond_degree_list_expanded = bond_degree_list_clamped.unsqueeze(-1).expand(batch_size, mol_length, max_neighbor_num, num_bond_features)
    
    # Step 3: Gather bond neighbors while preserving the 63 bonds dimension
    gathered_bond_neighbors = torch.gather(bond_list.unsqueeze(1).expand(batch_size, mol_length, num_bonds, num_bond_features), 2, bond_degree_list_expanded)
    
    # Now bond neighbors will have shape: [batch_size, mol_length, max_neighbor_num, num_bond_features]
    return gathered_bond_neighbors


class CustomminGRUCell(nn.Module):
    def __init__(self, hidden_features):
        super(CustomminGRUCell, self).__init__()
        self.hidden_features = hidden_features
        # Linear layers for input and hidden state transformation
        self.dense_i = nn.Linear(hidden_features, hidden_features)
        self.dense_h = nn.Linear(hidden_features, hidden_features)


    def forward(self, h, inputs):
        # GRU Gates
        htilde = self.dense_i(inputs)
        z = torch.sigmoid( self.dense_h(inputs))  # Update gate ie 0 to 1 value!
       
        # Update hidden state ie "(1-z) * h + z * htilde"
        new_h = torch.lerp(h, htilde, z) 
        return new_h  # Return updated hidden state


class CustomminLSTM(nn.Module):
    def __init__(self, hidden_features):
        super(CustomminLSTM, self).__init__()
        self.hidden_features = hidden_features

        self.linear_f = nn.Linear(hidden_features, hidden_features)
        self.linear_i = nn.Linear(hidden_features, hidden_features)
        self.linear_h = nn.Linear(hidden_features, hidden_features)

    def forward(self, h, inputs):
        # GRU Gates
        htilde = self.linear_h(inputs)
        # Compute gates
        f_t = torch.sigmoid(self.linear_f(inputs))  # Forget gate
        i_t = torch.sigmoid(self.linear_i(inputs))  # Input gate
         
        # Normalize gates
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)  
        # Update hidden state ie "(1-z) * h + z * htilde"
            
        new_h = f_prime_t * h + i_prime_t * htilde  
        
        return new_h  # Return updated hidden state



class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, output_units_num, p_dropout, minGRU=False):
        super(Fingerprint, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)
        
        if minGRU:
            self.GRUCell = nn.ModuleList([CustomminGRUCell(fingerprint_dim) for r in range(radius)])
        else:
            self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
            
        self.align = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        
        if minGRU:
            self.mol_GRUCell = CustomminGRUCell(fingerprint_dim)
        else:
            self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear1 = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.linear2 = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.output = nn.Linear(fingerprint_dim, output_units_num)
        self.scaler = nn.Linear(output_units_num, output_units_num)

        self.radius = radius
        self.T = T

  
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        # Ensure all inputs are moved to the same device as the model
        device = atom_list.device
        
        atom_mask = atom_mask.unsqueeze(2).to(device)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        
        atom_feature = F.leaky_relu(self.atom_fc(atom_list.to(device)))

        
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0).to(device)
        
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0).to(device)

        # Efficient neighbor gathering
        
        #atom_neighbor = gather_neighbors(atom_list, atom_degree_list)
        #bond_neighbor = gather_neighbors_bonds(bond_list, bond_degree_list)
        
    
        
        # Concatenate the atom and bond neighbors
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature.to(device)))
    
        # Generate mask to eliminate the influence of blank atoms
        attend_mask = torch.where(atom_degree_list != mol_length - 1, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        attend_mask = attend_mask.unsqueeze(-1)
        
        softmax_mask = torch.where(atom_degree_list != mol_length - 1, torch.tensor(0.0, device=device), torch.tensor(-9e8, device=device))
        softmax_mask = softmax_mask.unsqueeze(-1)        

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)
    
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask  # Ensure both tensors are on the same device
        attention_weight = F.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask
        
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = torch.sum(attention_weight * neighbor_feature_transform, dim=-2)
        context = F.elu(context)
    
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
    
        
        # Apply nonlinearity
        activated_features = F.leaky_relu(atom_feature)

        # Further steps through the radius
        for d in range(self.radius - 1):
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            attention_weight = attention_weight * attend_mask

            neighbor_feature_transform = self.attend[d + 1](self.dropout(neighbor_feature))
            context = torch.sum(attention_weight * neighbor_feature_transform, -2)
            context = F.elu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d + 1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            activated_features = F.leaky_relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2).to(device)

        activated_features_mol = F.leaky_relu(mol_feature)

        # Molecule-level attention and GRU
        mol_softmax_mask = torch.where(atom_mask == 0, torch.tensor(-9e8, device=device), torch.tensor(0.0, device=device))

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask

            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(mol_attention_weight * activated_features_transform, -2)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            activated_features_mol = F.leaky_relu(mol_feature)

        r0 =  self.dropout(mol_feature)  # added first activation without test rmse 0.5 and with 0.52
        r1 =  F.leaky_relu(self.linear1(r0))+r0
        r2 =  F.leaky_relu(self.linear2(self.dropout(r1)))+r1
        mol_prediction = self.scaler (self.output(r2))

        return atom_feature, mol_prediction
