import pandas as pd
import os

def generate_splits(csv_path, output_dir):
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(5):
        fold_col = f'Fold {i}'
        
        # Create boolean splits file
        bool_df = pd.DataFrame({
            '': data['case_id'],
            'train': data[fold_col] == 'train',
            'val': data[fold_col] == 'val',
            'test': data[fold_col] == 'val'  # Copy 'val' values to 'test'
        })
        bool_df.to_csv(os.path.join(output_dir, f'splits_{i}_bool.csv'), index=False)
        
        # Create list splits file
        train_ids = data['case_id'][data[fold_col] == 'train'].tolist()
        val_ids = data['case_id'][data[fold_col] == 'val'].tolist()
        test_ids = val_ids[:]  # Copy 'val' values to 'test'

        # Find maximum length of lists
        max_len = max(len(train_ids), len(val_ids), len(test_ids))

        # Ensure all columns have the same length by padding with empty strings
        train_ids.extend([''] * (max_len - len(train_ids)))
        val_ids.extend([''] * (max_len - len(val_ids)))
        test_ids.extend([''] * (max_len - len(test_ids)))

        list_df = pd.DataFrame({
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        })

        list_df.to_csv(os.path.join(output_dir, f'splits_{i}.csv'), index=False)


# Set the path to your CSV file and the output directory
csv_path = '/storage/Pathology/codes/EasyMIL/dataset_csv/survival_by_case/TCGA_LIHC_Splits.csv'
output_dir = '/storage/Pathology/codes/EasyMIL/splits82/TCGA_LIHC_survival_100'

generate_splits(csv_path, output_dir)
