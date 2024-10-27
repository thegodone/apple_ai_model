import tensorflow as tf
from tensorflow.keras import layers
import math
import os

def gather_neighbors_bonds(bond_list, bond_degree_list):
    """
    Efficiently gathers bond neighbor features using the bond degree list.
    Args:
    - `bond_list`: Tensor of shape [batch_size, num_bonds, num_bond_features]
    - `bond_degree_list`: Tensor of shape [batch_size, mol_length, max_neighbor_num]
    
    Returns:
    - Gathered bond neighbor features of shape [batch_size, mol_length, max_neighbor_num, num_bond_features]
    """
    bond_degree_list_clamped = tf.clip_by_value(bond_degree_list, 0, bond_list.shape[1] - 1)
    gathered_bond_neighbors = tf.gather(bond_list, bond_degree_list_clamped, batch_dims=1) 
    return gathered_bond_neighbors

class Fingerprint(tf.keras.Model):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint, self).__init__()
        self.radius = radius
        self.T = T

        self.atom_fc = layers.Dense(fingerprint_dim,name='atom_fc')
        self.neighbor_fc = layers.Dense(fingerprint_dim,name='neighbor_fc')
        self.GRUCell = [layers.GRUCell(fingerprint_dim, name='atomgru_%i'%r) for r in range(radius)]
        self.align = [layers.Dense(1,name='align_%i'%r) for r in range(radius)]
        self.attend = [layers.Dense(fingerprint_dim,name='attend_%i'%r) for r in range(radius)]

        self.mol_GRUCell = layers.GRUCell(fingerprint_dim, name='molgru')
        self.mol_align = layers.Dense(1,name='molalign')
        self.mol_attend = layers.Dense(fingerprint_dim,name='molattend')

        self.dropout = layers.Dropout(p_dropout,name='dropout')
        self.linear1 = layers.Dense(fingerprint_dim,name='linear1')
        self.linear2 = layers.Dense(fingerprint_dim,name='linear2')
        self.scalar = layers.Dense(output_units_num,name='scalar')
        self.output_layer = layers.Dense(output_units_num, name='output')

    def call(self, inputs, doprint=False):
        atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask = inputs
        atom_mask = tf.expand_dims(atom_mask, -1)
        batch_size, mol_length, num_atom_feat = atom_list.shape

        # Initial atom embedding
        atom_feature = tf.nn.leaky_relu(self.atom_fc(atom_list))
        if doprint: print("atom_feature shape:", atom_feature.shape)

        atom_neighbor = tf.gather(atom_list, atom_degree_list, batch_dims=1)
        bond_neighbor = gather_neighbors_bonds(bond_list, bond_degree_list)
        
        # Debug: print shapes of gathered neighbors
        if doprint: 
            print("atom_neighbor shape:", atom_neighbor.shape)  # Expected: [batch_size, mol_length, max_neighbor_num, atom_feature_dim]
            print("bond_neighbor shape:", bond_neighbor.shape)  # Expected: [batch_size, mol_length, max_neighbor_num, bond_feature_dim]
        
        neighbor_feature = tf.concat([atom_neighbor, bond_neighbor], axis=-1)
        if doprint: print("Concatenated neighbor_feature shape:", neighbor_feature.shape)  # Expected: [batch_size, mol_length, max_neighbor_num, atom_feature_dim + bond_feature_dim]

        neighbor_feature = tf.nn.leaky_relu(self.neighbor_fc(neighbor_feature))
        if doprint: print("Processed neighbor_feature shape:", neighbor_feature.shape)

        attend_mask = tf.where(atom_degree_list == mol_length - 1, 0.0, 1.0)
        attend_mask = tf.expand_dims(attend_mask, -1)
        
        softmax_mask = tf.where(atom_degree_list == mol_length - 1, -9e8, 0.0)
        softmax_mask = tf.expand_dims(softmax_mask, -1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = tf.expand_dims(atom_feature, axis=-2)
        atom_feature_expand = tf.tile(atom_feature_expand, [1, 1, max_neighbor_num, 1])
        
        feature_align = tf.concat([atom_feature_expand, neighbor_feature], axis=-1)
        if doprint: print("feature_align shape:", feature_align.shape)

        align_score = tf.nn.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score += softmax_mask
        if doprint: print("align_score shape:", align_score.shape)

        # Softmax over neighbors
        attention_weight = tf.nn.softmax(align_score, axis=-2)
        attention_weight = attention_weight * attend_mask
        if doprint: print("attention_weight shape:", attention_weight.shape)

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = tf.reduce_sum(attention_weight * neighbor_feature_transform, axis=-2)
        if doprint: print("context shape after reduce_sum:", context.shape)

        context_reshape = tf.reshape(context, [batch_size * mol_length, fingerprint_dim])
        atom_features_reshape = tf.reshape(atom_feature, [batch_size * mol_length, fingerprint_dim])
        atom_features_reshape, _ = self.GRUCell[0](context_reshape, [atom_features_reshape])
        if doprint: print("atom_features_reshape shape after GRU:", atom_features_reshape.shape)

        atom_features = tf.reshape(atom_features_reshape, [batch_size, mol_length, fingerprint_dim])
        activated_features = tf.nn.leaky_relu(atom_feature)
        
        # Iterate over the radius
        for r in range(self.radius - 1):
            neighbor_feature = tf.gather(activated_features, atom_degree_list, batch_dims=1)
            if doprint: print(f"neighbor_feature shape at radius {r}:", neighbor_feature.shape)

            atom_feature_expand = tf.expand_dims(atom_feature, axis=-2)
            atom_feature_expand = tf.tile(atom_feature_expand, [1, 1, max_neighbor_num, 1])
             
            feature_align = tf.concat([atom_feature_expand, neighbor_feature], axis=-1)
            if doprint: print(f"feature_align shape at radius {r}:", feature_align.shape)
    
            align_score = tf.nn.leaky_relu(self.align[r+1](self.dropout(feature_align)))
            align_score += softmax_mask
            if doprint: print(f"align_score shape at radius {r}:", align_score.shape)
    
            # Softmax over neighbors
            attention_weight = tf.nn.softmax(align_score, axis=-2)
            attention_weight = attention_weight * attend_mask
            if doprint: print(f"attention_weight shape at radius {r}:", attention_weight.shape)
    
            neighbor_feature_transform = self.attend[r+1](self.dropout(neighbor_feature))
            context = tf.reduce_sum(attention_weight * neighbor_feature_transform, axis=-2)
            if doprint: print(f"context shape after reduce_sum at radius {r}:", context.shape)
    
            context_reshape = tf.reshape(context, [batch_size * mol_length, fingerprint_dim])
            atom_features_reshape = tf.reshape(atom_feature, [batch_size * mol_length, fingerprint_dim])
            atom_features_reshape, _ = self.GRUCell[r+1](context_reshape, [atom_features_reshape])
    
            atom_features = tf.reshape(atom_features_reshape, [batch_size, mol_length, fingerprint_dim])
            activated_features = tf.nn.leaky_relu(atom_feature)
  
        # Mask and sum up atom features for the molecule
        mol_feature = tf.reduce_sum(atom_features * atom_mask, axis=-2)
        
        activated_features_mol = tf.nn.leaky_relu(mol_feature)

        # Molecule-level attention and GRU
        mol_softmax_mask = tf.where(atom_mask == 0, -9e8, 0.0)

        for t in range(self.T):
            mol_prediction_expand = tf.expand_dims(activated_features_mol, axis=-2)
            mol_prediction_expand = tf.tile(mol_prediction_expand, [1, mol_length, 1])

            mol_align = tf.concat([mol_prediction_expand, activated_features], axis=-1)
            mol_align_score = tf.nn.leaky_relu(self.mol_align(self.dropout(mol_align)))
            mol_align_score = mol_align_score + mol_softmax_mask

            mol_attention_weight = tf.nn.softmax(mol_align_score, axis=-2)
            mol_attention_weight = mol_attention_weight * atom_mask

            atom_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = tf.reduce_sum(mol_attention_weight * atom_features_transform, axis=-2)
            mol_context = tf.nn.elu(mol_context)

            mol_feature, _ = self.mol_GRUCell(mol_context, [mol_feature])
            activated_features_mol = tf.nn.leaky_relu(mol_feature)

        r0 = self.dropout(mol_feature)
        r1 = tf.nn.leaky_relu(self.linear1(r0)) + r0
        r2 = tf.nn.leaky_relu(self.linear2(self.dropout(r1))) + r1
        mol_prediction = self.scalar(self.output_layer(r2))

        return mol_prediction

class CosineAnnealingLR_with_Restart(tf.keras.callbacks.Callback):
    def __init__(self, T_max, T_mult, eta_min=0, verbose=1, model=None, out_dir="./", lr_reduction_factor=1.):
        super().__init__()
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.model = model
        self.out_dir = out_dir
        self.verbose = verbose
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_history = []
        self.current_epoch = 0
        self.base_lr = None

    def on_train_begin(self, logs=None):
        try:
            self.base_lr = self.model.optimizer.lr.numpy()
        except:
            self.base_lr = self.model.optimizer.learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        self.lr_history.append(lr)
        self.current_epoch += 1

        if self.verbose:
            print('\nEpoch %05d: CosineAnnealing lr to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        if self.current_epoch == self.Te:
            if self.verbose:
                print("Restart at epoch {:05d}".format(epoch + 1))

            if self.model is not None and self.out_dir:
                model_path = os.path.join(self.out_dir, "snapshot_e_{:05d}.keras".format(epoch + 1))
                self.model.save(model_path)

            # Reduce the base_lr by the reduction factor at each restart
            self.base_lr *= self.lr_reduction_factor
            self.current_epoch = 0
            self.Te = int(self.Te * self.T_mult)

