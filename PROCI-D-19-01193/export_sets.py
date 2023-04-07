from ecnet.utils.data_utils import DataFrame
from csv import DictWriter


def main():

    df = DataFrame('ysi_model_v3.0.csv')
    train_mols = [m for m in df.data_points if m.assignment == 'L']
    train_mols = sorted(train_mols, key=lambda m: float(m.TARGET))
    valid_mols = [m for m in df.data_points if m.assignment == 'V']
    valid_mols = sorted(valid_mols, key=lambda m: float(m.TARGET))
    test_mols = [m for m in df.data_points if m.assignment == 'T']
    test_mols = sorted(test_mols, key=lambda m: float(m.TARGET))

    rows = []
    for mol in train_mols:
        rows.append({
            'Compound Name': getattr(mol, 'Compound Name'),
            'SMILES': getattr(mol, 'SMILES'),
            'Experimental YSI': getattr(mol, 'TARGET')
        })
    with open('ysi_training_data.csv', 'w') as csv_file:
        writer = DictWriter(csv_file, ['Compound Name', 'SMILES', 'Experimental YSI'], delimiter=',', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    rows = []
    for mol in valid_mols:
        rows.append({
            'Compound Name': getattr(mol, 'Compound Name'),
            'SMILES': getattr(mol, 'SMILES'),
            'Experimental YSI': getattr(mol, 'TARGET')
        })
    with open('ysi_validation_data.csv', 'w') as csv_file:
        writer = DictWriter(csv_file, ['Compound Name', 'SMILES', 'Experimental YSI'], delimiter=',', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    rows = []
    for mol in test_mols:
        rows.append({
            'Compound Name': getattr(mol, 'Compound Name'),
            'SMILES': getattr(mol, 'SMILES'),
            'Experimental YSI': getattr(mol, 'TARGET')
        })
    with open('ysi_testing_data.csv', 'w') as csv_file:
        writer = DictWriter(csv_file, ['Compound Name', 'SMILES', 'Experimental YSI'], delimiter=',', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':

    main()
