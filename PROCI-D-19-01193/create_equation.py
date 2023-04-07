from ecnet.utils.data_utils import DataFrame
from scipy.optimize import curve_fit
from numpy import asarray, sqrt, diag


def exponential_equation(X, a, b, c):

    return X[0]**a + X[1]**b + X[2]**c


def main():

    df = DataFrame('ysi_model_v3.0.csv')
    mols = [m for m in df.data_points if m.assignment == 'L' or m.assignment == 'V']
    Y = asarray([float(mol.TARGET) for mol in mols])
    X = asarray([
        [float(mol.piPC05) for mol in mols],
        [float(mol.piPC04) for mol in mols],
        [float(mol.piPC03) for mol in mols]
    ])

    coef, cov = curve_fit(exponential_equation, X, Y)
    print('Coefficients: c5 = {}; c4 = {}; c3 = {}'.format(coef[0], coef[1], coef[2]))
    print(cov)
    print(sqrt(diag(cov)))


if __name__ == '__main__':

    main()
