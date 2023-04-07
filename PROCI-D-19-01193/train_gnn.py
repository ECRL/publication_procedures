import numpy as np

from preprocessor_class import atom_features, bond_features,\
    SmilesBondIndexPreprocessor

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Dense, BatchNormalization, Dropout,\
    Add, Concatenate, Multiply

from nfp.layers import ReduceAtomOrBondToMol, Squeeze, GatherAtomToBond,\
    ReduceBondToAtom, ReduceAtomOrBondToMol
from nfp.models import GraphModel
import warnings
from nfp.preprocessing import GraphSequence

from ecnet.utils.data_utils import DataFrame
from ecnet.utils.error_utils import calc_med_abs_error

from plot_results import plot

from ecnet.utils.logging import logger

import pandas as pd


def message_block(original_atom_state, original_bond_state, connectivity, i):

    atom_state = BatchNormalization()(original_atom_state)
    bond_state = BatchNormalization()(original_bond_state)

    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    new_bond_state = Concatenate(name='concat_{}'.format(i))([
        source_atom, target_atom, bond_state])
    new_bond_state = Dense(
        2*64, activation='relu')(new_bond_state)
    new_bond_state = Dense(64)(new_bond_state)

    bond_state = Add()([original_bond_state, new_bond_state])

    # message function
    source_atom = Dense(64)(source_atom)
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])

    # state transition function
    messages = Dense(64, activation='relu')(messages)
    messages = Dense(64)(messages)

    atom_state = Add()([original_atom_state, messages])

    return atom_state, bond_state


def main(fig_num):

    # Load database w/ explicit set assignments
    df = DataFrame('ysi_model_v3.0.csv')
    train_mols = [mol for mol in df.data_points if mol.assignment == 'L']
    valid_mols = [mol for mol in df.data_points if mol.assignment == 'V']
    test_mols = [mol for mol in df.data_points if mol.assignment == 'T']

    # Define/train preprocessor, transform data
    preprocessor = SmilesBondIndexPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features
    )
    inputs_train = preprocessor.fit([mol.SMILES for mol in train_mols])
    inputs_valid = preprocessor.predict([mol.SMILES for mol in valid_mols])
    inputs_test = preprocessor.predict([mol.SMILES for mol in test_mols])

    # Define target data
    targets_train = np.asarray([[float(mol.TARGET)] for mol in train_mols])
    targets_valid = np.asarray([[float(mol.TARGET)] for mol in valid_mols])
    targets_test = np.asarray([[float(mol.TARGET)] for mol in test_mols])

    num_messages = 3

    mol_type = Input(shape=(1,), name='n_atom', dtype='int32')
    node_graph_indices = Input(shape=(1,), name='node_graph_indices',
                               dtype='int32')
    bond_graph_indices = Input(shape=(1,), name='bond_graph_indices',
                               dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    squeeze = Squeeze()

    snode_graph_indices = squeeze(node_graph_indices)
    sbond_graph_indices = squeeze(bond_graph_indices)
    smol_type = squeeze(mol_type)
    satom_types = squeeze(atom_types)
    sbond_types = squeeze(bond_types)

    atom_state = Embedding(
        preprocessor.atom_classes, 64,
        name='atom_embedding'
    )(satom_types)
    bond_state = Embedding(
        preprocessor.bond_classes, 64,
        name='bond_embedding'
    )(sbond_types)
    bond_mean = Embedding(
        preprocessor.bond_classes, 1,
        name='bondwise_mean'
    )(sbond_types)

    for i in range(num_messages):
        atom_state, bond_state = message_block(
            atom_state, bond_state, connectivity, i
        )

    bond_state = Dropout(0.5)(bond_state)
    bond_state = Dense(1)(bond_state)
    bond_state = Add()([bond_state, bond_mean])
    mol_state = ReduceAtomOrBondToMol(reducer='sum')(
        [bond_state, sbond_graph_indices]
    )

    symb_inputs = [mol_type, node_graph_indices, bond_graph_indices,
                   atom_types, bond_types, connectivity]

    model = GraphModel(symb_inputs, [mol_state])

    batch_size = 32

    train_sequence = GraphSequence(inputs_train, targets_train, batch_size,
                                   final_batch=False)
    valid_sequence = GraphSequence(inputs_valid, targets_valid, batch_size,
                                   final_batch=False)
    epochs = 2000
    lr = 1E-3
    decay = 1E-5
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, decay=decay),
                  loss='msle')

    logger.log('info', 'Training GNN {}...'.format(fig_num))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hist = model.fit_generator(
            train_sequence,
            validation_data=valid_sequence,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=250,
                verbose=1,
                mode='min',
                restore_best_weights=True
            )],
            epochs=epochs,
            verbose=0
        )
        logger.log('info', 'Training complete after {} epochs'.format(len(
            hist.history['loss']
        )))

    train_predicted_values = model.predict_generator(GraphSequence(
        inputs_train,
        batch_size=batch_size,
        final_batch=True,
        shuffle=False
    ))
    valid_predicted_values = model.predict_generator(GraphSequence(
        inputs_valid,
        batch_size=batch_size,
        final_batch=True,
        shuffle=False
    ))
    test_predicted_values = model.predict_generator(GraphSequence(
        inputs_test,
        batch_size=batch_size,
        final_batch=True,
        shuffle=False
    ))

    mae_train = calc_med_abs_error(
        train_predicted_values,
        targets_train
    )
    mae_valid = calc_med_abs_error(
        valid_predicted_values,
        targets_valid
    )
    mae_test = calc_med_abs_error(
        test_predicted_values,
        targets_test
    )

    logger.log('info', 'Training med. abs. error: {}'.format(mae_train))
    logger.log('info', 'Validation med. abs. error: {}'.format(mae_valid))
    logger.log('info', 'Test med. abs. error: {}'.format(mae_test))

    # plot(
    #     train_predicted_values, targets_train,
    #     valid_predicted_values, targets_valid,
    #     test_predicted_values, targets_test,
    #     mae_train, mae_valid, mae_test,
    #     'figures\\gnn_{}'.format(fig_num)
    # )

    new_smiles = [
        'C=COCC1CCC(CC1)COC=C',
        'CCCCCCCC1CCC(=O)O1',
        'CCCCCCCCCC(=O)OCC',
        'CCCCCCCCCC(=O)OCCCC'
    ]
    inputs_new = preprocessor.fit(new_smiles)
    new_predicted_values = model.predict_generator(GraphSequence(
        inputs_new,
        batch_size=batch_size,
        final_batch=True,
        shuffle=False
    ))
    for i in range(len(new_smiles)):
        logger.log('info', 'Prediction for {}:\t {}'.format(
            new_smiles[i], new_predicted_values[i]
        ))

    bond_contribution_model = GraphModel(symb_inputs, [model.layers[-1].input[0]])
    train_merged_inputs = GraphSequence(inputs_train, y=None, batch_size=10000, final_batch=True, shuffle=False)
    train_predicted_bond_values = bond_contribution_model.predict_generator(train_merged_inputs)
    valid_merged_inputs = GraphSequence(inputs_valid, y=None, batch_size=10000, final_batch=True, shuffle=False)
    valid_predicted_bond_values = bond_contribution_model.predict_generator(valid_merged_inputs)
    test_merged_inputs = GraphSequence(preprocessor.predict([mol.SMILES for mol in test_mols]), y=None, batch_size=10000, final_batch=True, shuffle=False)
    test_predicted_bond_values = bond_contribution_model.predict_generator(test_merged_inputs)
    new_merged_inputs = GraphSequence(preprocessor.predict(new_smiles), y=None, batch_size=10000, final_batch=True, shuffle=False)
    new_predicted_bond_values = bond_contribution_model.predict_generator(new_merged_inputs)

    def inputs_to_dataframe(smiles, inputs):
        molecule = np.repeat(np.array(smiles), np.stack([iinput['n_bond'] for iinput in inputs]))
        bond_index = np.concatenate([iinput['bond_indices'] for iinput in inputs])
        input_df = pd.DataFrame(np.vstack([molecule, bond_index]).T,
                                columns=['molecule', 'bond_index'])
        input_df['bond_index'] = input_df.bond_index.astype('int64')
        return input_df

    train_df = inputs_to_dataframe([mol.SMILES for mol in train_mols], preprocessor.predict([mol.SMILES for mol in train_mols]))
    valid_df = inputs_to_dataframe([mol.SMILES for mol in valid_mols], preprocessor.predict([mol.SMILES for mol in valid_mols]))
    test_df = inputs_to_dataframe([mol.SMILES for mol in test_mols], preprocessor.predict([mol.SMILES for mol in test_mols]))
    new_df = inputs_to_dataframe(new_smiles, preprocessor.predict(new_smiles))
    all_df = pd.concat([train_df, valid_df, test_df, new_df])
    all_df['ysi_contribution'] = np.vstack([train_predicted_bond_values, valid_predicted_bond_values, test_predicted_bond_values, new_predicted_bond_values])
    bond_ysis = all_df.groupby(['molecule', 'bond_index']).mean().reset_index()
    bond_ysis.to_csv('predicted_ysi_contributions.csv', index=False)


if __name__ == '__main__':

    logger.stream_level = 'debug'
    logger.log_dir = 'logs_gnn'
    logger.file_level = 'debug'
    main(1)
    # for i in range(10):
    #     main(i)
