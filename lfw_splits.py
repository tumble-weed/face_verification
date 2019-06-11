import os
import numpy as np
import shutil
import pickle
import sys
def create_splits(classes,train_ratio):
    n_total_classes = len(classes)
    n_train_classes = int(train_ratio * n_total_classes)
    shuffled_class_order = np.arange(len(classes))
    np.random.shuffle(shuffled_class_order)
    shuffled_classes = np.array(classes)[shuffled_class_order]
    train_classes = shuffled_classes[:n_train_classes]
    test_classes = shuffled_classes[n_train_classes:]
    splits = {'train':train_classes,'test':test_classes}

    with open('lfw_splits','wb') as f:
        pickle.dump(splits,f)
    return splits

def create_soft_dir(root_folder,soft_dir,splits):
    if os.path.isdir(soft_dir):
        shutil.rmtree(soft_dir)
    os.makedirs(soft_dir)
    train_dir = os.path.join(soft_dir,'train')
    test_dir = os.path.join(soft_dir,'test')

    os.makedirs(train_dir)
    for d in splits['train']:
        os.system(f'ln -s {os.path.join(root_folder,d)} {os.path.join(train_dir,d)}')

    os.makedirs(test_dir)
    for d in splits['test']:
        os.system(f'ln -s {os.path.join(root_folder,d)} {os.path.join(test_dir,d)}')
    pass

def main():
    if len(sys.argv) > 1:
        splits_file = sys.argv[1]
        with open(splits_file,'rb') as f:
           splits = pickle.load(f)
    else:
        folder_structure = os.listdir('lfw')
        splits = create_splits(folder_structure,0.8)
    create_soft_dir('lfw','lfw_divided',splits)
    pass

if __name__ == '__main__':
    main()
    pass
