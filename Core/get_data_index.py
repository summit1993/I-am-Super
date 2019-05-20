import os
import pickle

def get_dat_index(root, dir_name, index_list, has_high):
    low_dir = dir_name + '_l'
    for _, dirs, _ in os.walk(os.path.join(root), low_dir):
        for dir in dirs:
            abs_dir = os.path.join(root, low_dir, dir)
            for _, _, files in os.walk(abs_dir):
                file_nums = len(files)
                print(file_nums)
                for file in files:
                    tmp_dict = {}
                    tmp_dict['image'] = file
                    tmp_dict['nums'] = file_nums
                    tmp_dict['low_dir'] = os.path.join(low_dir, dir)
                    if has_high:
                        high_dir = dir_name + '_h_GT'
                        tmp_dict['high_dir'] = os.path.join(high_dir, dir)
                    index_list.append(tmp_dict)
    return index_list

if __name__ == '__main__':
    root_dir = 'images'
    train_index = []
    val_index = []
    train_names = ['youku_00000_00049', 'youku_00050_00099', 'youku_00100_00149']
    val_names = ['youku_00150_00199']

    for name in train_names:
        train_index = get_dat_index(root_dir, name, train_index, True)

    for name in val_names:
        val_index = get_dat_index(root_dir, name, val_index, True)

    pickle.dump(train_index, open('train.pkl', 'wb'))
    pickle.dump(val_index, open('val.pkl', 'wb'))

