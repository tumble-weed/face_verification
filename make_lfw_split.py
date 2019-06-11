import numpy as np
import pickle
'''
Getting information about the lfw folder
'''
#-------------------------------------------------------------------------------
''' get the folder structure '''
import os,collections
train_folder = 'lfw'
folder_structure = collections.OrderedDict({})
classes = [d for d in sorted(os.listdir(train_folder)) if os.path.isdir(os.path.join(train_folder,d)) and d not in ['.','..']]
# if test_mode:
# #     classes = [classes[c] for c in [2,5]]
#     classes = classes[:10]
print(len(classes))
folder_structure = collections.OrderedDict({c:[f for f in os.listdir(os.path.join(train_folder,c)) if not os.path.isdir(f)] for c in classes})
if False:print(folder_structure)
#-------------------------------------------------------------------------------
''' which classes have which files '''
class_to_idx = {k:[] for k in folder_structure.keys()}
filelist = []
i = 0
for ( c ,fls) in folder_structure.items():   
    
    for fi in fls:
        filelist.append('/'.join([c,fi]))
        class_to_idx[c].append(i)
        i+=1

##
train_ratio = 0.8

new_class_order = np.random.permutation(range(len(classes)))
n_train_classes = int(len(classes)*train_ratio)
train_idx = new_class_order[:n_train_classes]
test_idx = new_class_order[n_train_classes:]

splits ={'train':train_idx,'test':test_idx}
with open('lfw_splits','wb') as f:
  pickle.dump(splits,f)



