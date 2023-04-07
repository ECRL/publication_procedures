from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def plot(y_train, t_train, y_valid, t_valid, y_test, t_test,
         er_train, er_valid, er_test, fig_name):

    plt.clf()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.yscale('log')
    plt.ylim([5, 2000])
    plt.xlim([5, 2000])
    plt.xscale('log')
    plt.scatter(t_train, y_train, label='Training', c='blue')
    plt.scatter(t_valid, y_valid, label='Validation', c='green')
    plt.scatter(t_test, y_test, label='Testing', c='red')
    plt.legend(loc=2, edgecolor='w')
    plt.plot([5, 2000], [5, 2000], color='k', linestyle=':', linewidth=1,
             zorder=0)
    text_box = AnchoredText('MAE (Training): %.2f\nMAE (Validation): '
                            '%.2f\nMAE (Testing): %.2f' % (
                                er_train, er_valid, er_test
                            ), frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', edgecolor='w')
    plt.gca().add_artist(text_box)
    plt.xlabel('YSI (Experimental)')
    plt.ylabel('YSI (Predicted)')
    plt.savefig('{}.png'.format(fig_name))
