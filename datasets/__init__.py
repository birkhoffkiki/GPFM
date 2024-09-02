from .dataset_generic import Generic_MIL_Dataset
from .dataset_survival import Generic_MIL_Survival_Dataset
import os


def get_survival_dataset(task, seed=119, data_root_dir = None):
    study = '_'.join(task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_lung'
    else:
        combined_study = study
    # combined_study = combined_study.split('_')[1]
    csv_path = 'dataset_csv/survival_by_case/{}_Splits.csv'.format(combined_study)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    
    # dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/%s_processed.csv' % combined_study,
    print(csv_path)
    dataset = Generic_MIL_Survival_Dataset(csv_path = csv_path,
                                            data_dir= data_root_dir,
                                            shuffle = False, 
                                            seed = seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
    return dataset


def get_subtying_dataset(task, seed=119, data_dir=None):
    if task == 'LUAD_LUSC':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                                data_dir= data_dir,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LUSC':1},
                                patient_strat=False,
                                ignore=[])

    elif task == 'camelyon':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'normal':0, 'tumor':1},
                                patient_strat= False,
                                ignore=[])
        
    elif task == 'RCC':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/RCC.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'KICH':0, 'KIRP':1, 'KIRC':2},
                                patient_strat= False,
                                ignore=[])

    elif task == 'PANDA':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/PANDA.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'0':0, '1':1, '2':2, '3':3, '4': 4, '5':5},
                                patient_strat=False,
                                ignore=[])
    elif task == 'BRACS':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRACS.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'PB':0, 'IC':1, 'DCIS':2, 'N':3, 'ADH': 4,
                                            'FEA':5, 'UDH': 6 },
                                patient_strat=False,
                                ignore=[])

    elif task == 'BRACS-3':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRACS-3.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Benign':0, 'AT':1, 'MT':2},
                                patient_strat=False,
                                ignore=[])

    elif task == 'LUAD_LUSC_STAD':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC_STAD.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LUSC':1, 'STAD':2 },
                                patient_strat=False,
                                ignore=[])
        
    elif task == 'TCGA_BRCA_subtyping':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRCA_subtyping.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'ILC':0, 'IDC':1},
                                patient_strat=False,
                                ignore=[])

    elif task == 'TCGA_BRCA_molecular_subtyping':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRCA_molecular_subtyping.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Normal':0, 'LumA':1, 'LumB':2, 'Basal':3, 'Her2':4},
                                patient_strat=False,
                                ignore=[])
        
    elif task == 'TCGA_COAD_READ_molecular_subtyping':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/COAD_READ_molecular_subtyping.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'CMS1':0, 'CMS2':1, 'CMS3':2, 'CMS4':3},
                                patient_strat=False,
                                ignore=[])
    
    elif task == 'UBC-OCEAN':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/UBC-OCEAN.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'CC': 0, 'HGSC': 1, 'LGSC': 2, 'EC': 3, 'MC': 4},
                                patient_strat=False,
                                ignore=[])
        
    elif task == 'CPTAC_LUAD_LSCC':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CPTAC_LUAD_LSCC.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LSCC':1},
                                patient_strat=False,
                                ignore=[])
        
    elif task == 'CPTAC_LUAD_LSCC_Normal':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CPTAC_LUAD_LSCC_Normal.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'luad':0, 'lscc':1, 'normal':2},
                                patient_strat=False,
                                ignore=[])
        
    elif task == 'TUPAC16':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TUPAC16.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "TCGA_GBMLGG_IDH1":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/GBMLGG_IDH1.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'0':0, '1':1},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "BCNB_ER":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BCNB_ER.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Negative':0, 'Positive':1},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "BCNB_PR":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BCNB_PR.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Negative':0, 'Positive':1},
                                patient_strat=False,
                                ignore=[])

    elif task == "BCNB_HER2":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BCNB_HER2.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Negative':0, 'Positive':1},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "TCGA_LUAD_TP53":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_LUAD_TP53.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Negative':0, 'Positive':1},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "TCGA_LUAD_EGFR":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_LUAD_EGFR.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Negative':0, 'Positive':1},
                                patient_strat=False,
                                ignore=[])

    elif task == 'eBrains_32':
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/ebrains_fine.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'Anaplastic_ependymoma': 0, 'Fibrous_meningioma': 1, 'Anaplastic_astrocytoma__IDH-wildtype': 2,
                                            'Angiomatous_meningioma': 3, 'Secretory_meningioma': 4, 'Lipoma': 5, 'Pilocytic_astrocytoma': 6, 'Atypical_meningioma': 7, 'Anaplastic_oligodendroglioma__IDH-mutant_and_1p/19q_codeleted': 8,
                                            'Anaplastic_astrocytoma__IDH-mutant': 9, 'Diffuse_astrocytoma__IDH-mutant': 10, 'Diffuse_large_B-cell_lymphoma_of_the_CNS': 11,
                                            'Transitional_meningioma': 12,
                                            'Meningothelial_meningioma': 13,
                                            'Langerhans_cell_histiocytosis': 14,
                                            'Gliosarcoma': 15,
                                            'Ganglioglioma': 16,
                                            'Anaplastic_meningioma': 17,
                                            'Chordoma': 18,
                                            'Schwannoma': 19,
                                            'Adamantinomatous_craniopharyngioma': 20,
                                            'Medulloblastoma__non-WNT/non-SHH': 21,
                                            'Psammomatous_meningioma': 22,
                                            'Haemangioblastoma': 23,
                                            'Metastatic_tumours': 24,
                                            'Haemangioma': 25,
                                            'Haemangiopericytoma': 26,
                                            'Ependymoma': 27,
                                            'Glioblastoma__IDH-mutant': 28,
                                            'Pituitary_adenoma': 29,
                                            'Oligodendroglioma__IDH-mutant_and_1p/19q_codeleted': 30,
                                            'Glioblastoma__IDH-wildtype': 31},
                                patient_strat=False,
                                ignore=[])
        
    elif task == "TPM_binary":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TPM-binary.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'N':0, 'T':1},
                                patient_strat=False,
                                ignore=[])

    elif task == "TPM_multi_class":
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TPM-multi-class.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = seed, 
                                print_info = True,
                                label_dict = {'N':0, 'CIS':1, 'ILC':2, 'IDC': 3},
                                patient_strat=False,
                                ignore=[])
 
    else:
        raise NotImplementedError
    return dataset
        