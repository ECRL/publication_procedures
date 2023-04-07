import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

bond_ysis = pd.read_csv('predicted_ysi_contributions.csv')

norm = mpl.colors.Normalize(
    vmin=np.percentile(bond_ysis.ysi_contribution, 5),
    vmax=np.percentile(bond_ysis.ysi_contribution, 95))
cmap = cm.plasma

m = cm.ScalarMappable(norm=norm, cmap=cmap)
rgb2hex = lambda r, g, b: f"#{r:02x}{g:02x}{b:02x}"


def get_color(x):

    rgba = np.asarray(m.to_rgba(float(x)))
    return tuple(rgba[:-1])


def draw_mol_svg(smiles, figsize=(300, 300)):

    color_dict = pd.Series({i: get_color(x) for i, x in enumerate(bond_ysis[bond_ysis.molecule == smiles].ysi_contribution)})

    mol = Chem.MolFromSmiles(smiles)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)

    n_bonds = len(mol.GetBonds())
    assert n_bonds == len(color_dict), "{} bonds in mol, {} colors".format(n_bonds, len(color_dict))

    if color_dict is not None:

        drawer.DrawMolecule(
            mc, highlightBonds=list(color_dict.index),
            highlightBondColors=color_dict.to_dict(),
            highlightAtoms=False)

    else:
        drawer.DrawMolecule(mc)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '').replace(':svg', '')
    with open('bond_contr\\_{}.svg'.format(smiles), 'w') as imfile:
        imfile.write(svg)
    imfile.close()


def main():

    # plt.rcParams['font.family'] = 'Times New Roman'
    # bond_ysis.ysi_contribution.plot.hist(bins=30, color='blue')
    # plt.xlabel('Numerical Contribution to Unified YSI')
    # plt.show()
    draw_mol_svg('C1=CCCCC1')



if __name__ == '__main__':

    main()
