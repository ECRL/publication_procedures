from ecnet.utils.data_utils import DataFrame
from ecnet.utils.error_utils import calc_med_abs_error
from plot_results import plot
from numpy import asarray
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

c5 = 3.8851302931152727
c4 = 3.156729225555387e-06
c3 = 1.4326034483469652


def exponential_equation(X, a, b, c):

    return X[0]**a + X[1]**b + X[2]**c


def main():

    df = DataFrame('ysi_model_v3.0.csv')
    train_mols = [m for m in df.data_points if m.assignment == 'L']
    valid_mols = [m for m in df.data_points if m.assignment == 'V']
    test_mols = [m for m in df.data_points if m.assignment == 'T']

    train_x = asarray([
        [float(m.piPC05), float(m.piPC04), float(m.piPC03)]
        for m in train_mols
    ])
    train_y = asarray([float(m.TARGET) for m in train_mols])

    valid_x = asarray([
        [float(m.piPC05), float(m.piPC04), float(m.piPC03)]
        for m in valid_mols
    ])
    valid_y = asarray([float(m.TARGET) for m in valid_mols])

    test_x = asarray([
        [float(m.piPC05), float(m.piPC04), float(m.piPC03)]
        for m in test_mols
    ])
    test_y = asarray([float(m.TARGET) for m in test_mols])

    train_pred = [exponential_equation(x, c5, c4, c3) for x in train_x]
    valid_pred = [exponential_equation(x, c5, c4, c3) for x in valid_x]
    test_pred = [exponential_equation(x, c5, c4, c3) for x in test_x]

    train_mae = calc_med_abs_error(train_pred, train_y)
    valid_mae = calc_med_abs_error(valid_pred, valid_y)
    test_mae = calc_med_abs_error(test_pred, test_y)

    print('Training MAE: {}'.format(train_mae))
    print('Validation MAE: {}'.format(valid_mae))
    print('Testing MAE: {}'.format(test_mae))

    new_mols = DataFrame('new_mols.csv')
    new_x = asarray([
        [float(m.piPC05), float(m.piPC04), float(m.piPC03)]
        for m in new_mols.data_points
    ])

    preds = [exponential_equation(x, c5, c4, c3) for x in new_x]
    for i, p in enumerate(preds):
        print('Prediction for {}: {}'.format(new_mols.data_points[i].SMILES, p))

    # plot(train_pred, train_y, valid_pred, valid_y, test_pred, test_y,
    #      train_mae, valid_mae, test_mae, 'figures\\equation')

    plt.clf()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.yscale('log')
    plt.ylim([5, 2000])
    plt.xlim([5, 2000])
    plt.xscale('log')
    plt.scatter(train_y, train_pred, label='Training', c='blue')
    plt.scatter(valid_y, valid_pred, label='Validation', c='green')
    plt.scatter(test_y, test_pred, label='Testing', c='red')
    plt.legend(loc=2, edgecolor='w')
    plt.plot([5, 2000], [5, 2000], color='k', linestyle=':', linewidth=1,
             zorder=0)
    text_box = AnchoredText('MAE (Training): %.2f\nMAE (Validation): '
                            '%.2f\nMAE (Testing): %.2f' % (
                                train_mae, valid_mae, test_mae
                            ), frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', edgecolor='w')
    plt.gca().add_artist(text_box)
    plt.xlabel('YSI (Experimental)')
    plt.ylabel('YSI (Predicted)')
    plt.show()
    # plt.savefig('{}.png'.format('figures\\equation_parity'))


if __name__ == '__main__':

    main()
