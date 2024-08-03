# text="""BLCA
# BRCA
# CRC
# GBMLGG
# HNSC
# KIRC
# KIRP
# LIHC
# LUAD
# LUSC
# PAAD
# SKCM
# STAD
# UCEC"""

# items = text.split('\n')
# for ind, it in enumerate(items):
#     print('arr[{}]=\"{}\"'.format(ind, it))

import pickle

f = open('/jhcnas3/Pathology/experiments/train/LUAD_LUSC/att_mil/resnet50_s1/split_0_results.pkl', 'rb')
info = pickle.load(f)
print(info)