import mlx.core as mx
import math
import mlx.optimizers as optim
import mlx.nn as nn


class minGRUCell(nn.Module):
    """A minGRU Cell that returns the final hidden state only."""
    def __init__(
        self,
        inputs_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.linear_i = nn.Linear(inputs_size, hidden_size)
        self.linear_h = nn.Linear(inputs_size, hidden_size)

    def __call__(self, h, inputs):
        
        hidden = self.linear_i(inputs)
        
        z = mx.sigmoid(self.linear_h(inputs))

        hidden = (1 - z) * h + z * hidden
 
        return hidden



class GRUCell(nn.Module):
    """A GRU Cell that returns the final hidden state only."""
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def __call__(self, hx, inputs):

        x_t = self.x2h(inputs)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = mx.split(x_t,3, 1)
        h_reset, h_upd, h_new = mx.split(h_t,3, 1)

        reset_gate = mx.sigmoid(x_reset + h_reset)
        update_gate = mx.sigmoid(x_upd + h_upd)
        new_gate = mx.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class AttFP(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, output_units_num, p_dropout=0.1, printing=False):
        super(AttFP, self).__init__()
        
        self.atom_fc =  nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc =  nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)
        self.GRUCell = [GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        self.align = [nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)]
        self.attend = [nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        
        self.molGRU =  GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear1 = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.linear2 = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.output = nn.Linear(fingerprint_dim, output_units_num)
        self.scaler = nn.Linear(output_units_num,output_units_num)

        self.radius = radius
        self.T = T
        self.printing=printing

    def __call__(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):

      
        atom_mask = mx.expand_dims(atom_mask,2)
        batch_size, mol_length, num_atom_feat = atom_list.shape
        atom_feature = nn.leaky_relu(self.atom_fc(atom_list))

        #bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        #bond_neighbor = mx.stack(bond_neighbor, axis=0)       
        #atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        #atom_neighbor = mx.stack(atom_neighbor, axis=0)

        atom_neighbor = mx.take_along_axis(atom_list[..., None, :], atom_degree_list[..., None], axis=-3)
        bond_neighbor = mx.take_along_axis(bond_list[..., None, :], bond_degree_list[..., None], axis=-3)

        neighbor_feature = mx.concatenate([atom_neighbor, bond_neighbor], axis=-1)
        
        neighbor_feature = nn.leaky_relu(self.neighbor_fc(neighbor_feature))
        
        
        attend_mask = mx.where(atom_degree_list == mol_length - 1, mx.array(0.0), mx.array(1.0))
        attend_mask = mx.expand_dims(attend_mask,-1)
        
        softmax_mask = mx.where(atom_degree_list == mol_length - 1, mx.array(-9e8), mx.array(0.0))
        softmax_mask = mx.expand_dims(softmax_mask,-1)
        
        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
                
        atom_feature_expand = mx.expand_dims(atom_feature, -2) 
        atom_feature_expand = mx.repeat(atom_feature_expand,  max_neighbor_num, axis=-2)
        
        feature_align = mx.concatenate([atom_feature_expand, neighbor_feature], axis=-1)
        
        align_score = nn.leaky_relu(self.align[0](self.dropout(feature_align)))       
        align_score = align_score + softmax_mask
        attention_weight = nn.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask
        

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
       
        context = mx.sum(attention_weight * neighbor_feature_transform, axis=-2)
        context = nn.elu(context)
        
        context_reshape = mx.reshape(context, (batch_size * mol_length, fingerprint_dim))
            
        atom_feature_reshape = mx.reshape(atom_feature, (batch_size * mol_length, fingerprint_dim))

        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
       
        atom_feature = mx.reshape(atom_feature_reshape, (batch_size, mol_length, fingerprint_dim))
        
        activated_features = nn.leaky_relu(atom_feature)
        
        for d in range(self.radius-1):
            #neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            #neighbor_feature = mx.stack(neighbor_feature, axis=0)
            neighbor_feature = mx.take_along_axis(activated_features[..., None, :], atom_degree_list[..., None], axis=-3)
           
            atom_feature_expand = mx.expand_dims(activated_features, -2)
            atom_feature_expand = mx.repeat(atom_feature_expand, max_neighbor_num, axis=-2)
                
            feature_align = mx.concatenate([atom_feature_expand, neighbor_feature], axis=-1)
               
            align_score = nn.leaky_relu(self.align[d+1](self.dropout(feature_align)))
            align_score = align_score + softmax_mask
                
            attention_weight = nn.softmax(align_score, -2)
            attention_weight = attention_weight * attend_mask
            
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
            context = mx.sum(attention_weight * neighbor_feature_transform, axis=-2)
            context = nn.elu(context)
               
            context_reshape = mx.reshape(context, (batch_size * mol_length, fingerprint_dim))
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = mx.reshape(atom_feature_reshape, (batch_size, mol_length, fingerprint_dim))
            activated_features = nn.leaky_relu(atom_feature)
           
        mol_feature = mx.sum(activated_features * atom_mask, axis=-2)
            
        activated_features_mol = nn.leaky_relu(mol_feature)
            
        mol_softmax_mask = mx.where(atom_mask == 0,  mx.array(-9e8),mx.array(0.0))
            
        for t in range(self.T):
            mol_prediction_expand = mx.expand_dims(activated_features_mol, -2)
            mol_prediction_expand = mx.repeat(mol_prediction_expand, mol_length, axis=1)
            mol_align = mx.concatenate([mol_prediction_expand, activated_features], axis=-1)
             
            mol_align_score = nn.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
                
            mol_attention_weight = nn.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask
                
            activated_feature_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = mx.sum(mol_attention_weight * activated_feature_transform, axis=-2)
            mol_context = nn.elu(mol_context)
                
            mol_feature = self.molGRU(mol_context, mol_feature)
            activated_features_mol = nn.leaky_relu(mol_feature)
            
        r0 = self.dropout(mol_feature)
        r1 = nn.leaky_relu(self.linear1(r0))+r0
        r2 = nn.leaky_relu(self.linear1(self.dropout(r1)))+r1
        mol_prediction = self.scaler(self.output(r2))

        return atom_feature, mol_prediction


# Define a function that dynamically creates [cosine_i, warmup_i] schedules
def cosineannealingwarmrestartfactor(initial_lr, restart, decay_steps, warmup_factor):
    schedules = []
    boundaries = []  # Boundaries should be one less than schedules
    # Loop through each milestone and create a pair of cosine and warmup schedules
    schedules.append(optim.cosine_decay(initial_lr, decay_steps))
    for i in range(restart-1):        
        # Create cosine decay schedule for this phase
        initial_lr*=warmup_factor
        # Append the cosine and warmup schedules
        schedules.append(optim.cosine_decay(initial_lr, decay_steps))
        boundaries.append(decay_steps*(i+1))
    print(boundaries)
    # Combine the schedules dynamically based on milestones
    lr_schedule = optim.join_schedules(schedules, boundaries)
    return lr_schedule

if __name__ == '__main__':
    
    # Example: Dynamically create learning rate schedules based on milestones
    initial_lr = 1e-3
    restarts = 10
    decay_step = 20  # Decay steps for each cosine and warmup phase caution step = iteration not epochs!
    warmup_factor = 0.95  # Warmup reduction factors
    
    
    lr_schedule = cosineannealingwarmrestartfactor(initial_lr, restarts, decay_step, warmup_factor)
    
    # Use it in the optimizer
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    
    # Simulate optimizer updates and print learning rate values
    r = []
    for _ in range(80):  # Simulate 50 steps
        optimizer.update({}, {})
        r.append(optimizer.learning_rate)
    
    plt.plot(r);
