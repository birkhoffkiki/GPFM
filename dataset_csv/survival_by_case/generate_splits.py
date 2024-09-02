import pandas as pd
import os

def create_splits(input_csv):
    # Read the original csv file
    df = pd.read_csv(input_csv)

    # Extract the study name
    study_name = df['Study'].iloc[0].split('-')[-1]
    output_dir = f"/storage/Pathology/codes/EasyMIL/splits82/TCGA_{study_name}_survival_100"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold in range(5):
        fold_column = f"Fold {fold}"

        # Initialize lists to collect train, val, and test case_ids
        train_cases = []
        val_cases = []
        test_cases = []

        for idx, row in df.iterrows():
            case_id = row['case_id']
            fold_value = row[fold_column]
            if fold_value == "train":
                train_cases.append(case_id)
            elif fold_value == "val":
                val_cases.append(case_id)
                test_cases.append(case_id)
            elif fold_value == "test":
                test_cases.append(case_id)
                val_cases.append(case_id)

        # Create DataFrame for splits_{fold_number}.csv
        max_length = max(len(train_cases), len(val_cases), len(test_cases))
        splits_df = pd.DataFrame({
            'train': train_cases + [None] * (max_length - len(train_cases)),
            'val': val_cases + [None] * (max_length - len(val_cases)),
            'test': test_cases + [None] * (max_length - len(test_cases))
        })
        splits_df.index.name = 'index'
        splits_df.to_csv(f"{output_dir}/splits_{fold}.csv")

        # Create DataFrame for splits_{fold_number}_bool.csv
        bool_df = pd.DataFrame({
            'case_id': df['case_id'],
            'train': df[fold_column] == 'train',
            'val': (df[fold_column] == 'val') | (df[fold_column] == 'test'),
            'test': (df[fold_column] == 'test') | (df[fold_column] == 'val')
        })
        bool_df.to_csv(f"{output_dir}/splits_{fold}_bool.csv", index=False)

# Usage
if __name__ == "__main__":
    input_csv = '/storage/Pathology/codes/EasyMIL/dataset_csv/survival_by_case/TCGA_CESC_Splits.csv'
    create_splits(input_csv)
