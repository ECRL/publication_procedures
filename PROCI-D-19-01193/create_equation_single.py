from ecnet.utils.data_utils import DataFrame
from scipy.optimize import curve_fit
from numpy import asarray


def exponential_equation(X, a):

    return X[0]**a


def main():

    df = DataFrame('ysi_model_v3.0.csv')
    mols = [m for m in df.data_points if m.assignment == 'L' or m.assignment == 'V']
    Y = asarray([float(mol.TARGET) for mol in mols])
    X = asarray([
        [float(mol.piPC05) for mol in mols]
    ])

    coef, _ = curve_fit(exponential_equation, X, Y)
    print('Coefficients: c5 = {}'.format(coef[0]))


if __name__ == '__main__':

    main()
