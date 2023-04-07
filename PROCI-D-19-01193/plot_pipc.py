from matplotlib import pyplot as plt
from ecnet.utils.data_utils import DataFrame


def main():

    df = DataFrame('ysi_model_v3.0.csv')
    mols = [pt for pt in df.data_points if pt.assignment == 'L' or pt.assignment == 'V']
    ysi = [float(m.TARGET) for m in mols]
    pipc = [float(getattr(m, 'piID')) for m in mols]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel('piPC05 value')
    # plt.yscale('log')
    # plt.ylim([5, 1500])
    # plt.xscale('log')
    # plt.xlim([5, 1500])
    plt.ylabel('YSI (Experimental)')
    plt.scatter(pipc, ysi, s=[5 for _ in range(len(ysi))], c='blue')
    plt.show()


if __name__ == '__main__':

    main()
