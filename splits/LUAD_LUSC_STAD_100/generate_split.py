import pandas as pd

df_1 = pd.read_csv('/storage/Pathology/codes/CLAM/splits/LUAD_LUSC_STAD_100/splits_0.csv')
df_2 = pd.read_csv('/storage/Pathology/codes/CLAM/dataset_csv/LUAD_LUSC_STAD.csv')

def match_full_name(prefix):
    # Find the full name that starts with the prefix
    full_name = df_2['case_id'][df_2['case_id'].str.startswith(str(prefix))].values
    # Return the full name if found, otherwise return the prefix
    return full_name[0] if len(full_name) > 0 else prefix

# Apply the function to each element in df_1
df_1 = df_1.applymap(match_full_name)

df_1 = df_1.drop(columns=['Unnamed: 0'])

df_1.to_csv('/storage/Pathology/codes/CLAM/splits/LUAD_LUSC_STAD_100/splits.csv', index=True)