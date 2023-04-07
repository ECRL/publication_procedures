from ecnet import Server
from ecnet.utils.logging import logger
from plot_results import plot


def main(fig_num):

    logger.log('info', 'Training ANN {}...'.format(fig_num))
    sv = Server()
    sv._vars['hidden_layers'] = [
        [1500, 'relu'],
        [750, 'relu']
    ]
    sv.load_data('ysi_model_v3.0.csv')
    sv.train(validate=True)
    train_errors = sv.errors('med_abs_error', dset='learn')
    valid_errors = sv.errors('med_abs_error', dset='valid')
    test_errors = sv.errors('med_abs_error', dset='test')
    train_preds = sv.use('learn')
    valid_preds = sv.use('valid')
    test_preds = sv.use('test')

    plot(
        train_preds, sv._sets.learn_y,
        valid_preds, sv._sets.valid_y,
        test_preds, sv._sets.test_y,
        train_errors['med_abs_error'],
        valid_errors['med_abs_error'],
        test_errors['med_abs_error'],
        'figures\\ann_{}'.format(fig_num)
    )

    sv.load_data('new_mols.csv')
    new_mol_results = sv.use()
    for i in range(len(new_mol_results)):
        logger.log('info', 'Prediction for {}:\t {}'.format(
            sv._df.data_points[i].SMILES, new_mol_results[i]
        ))


if __name__ == '__main__':

    logger.stream_level = 'debug'
    logger.log_dir = 'logs_ann'
    logger.file_level = 'debug'
    for i in range(10):
        main(i)
