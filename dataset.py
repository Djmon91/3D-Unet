#coding:utf-8
'''
* @auther tzw
* @date 2018-6-15
'''
import os, sys, time
import numpy as np
import chainer
import util.dataIO as IO

class UnetDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, data_list_txt, patch_side, number_of_label, validation=None):
        print(' Initilaze dataset ')
        self._root = root
        self._patch_side = patch_side
        self._max_label = number_of_label #[0, number_of_label)
        self._validation = validation

        assert(self._patch_side%2==0)

        """
        * Read path to org and label data
        hogehoge.txt
        org.mhd org_label.mhd
        """
        path_pairs = []
        with open(data_list_txt) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                path_pairs.append(line[:])

        self._num_of_case = len(path_pairs)
        print('    # of cases: {}'.format(self._num_of_case))

        self._dataset=[]
        for i in path_pairs:
            print('   Org   from: {}'.format(i[0]))
            print('   label from: {}'.format(i[1]))
            print('   liver mask from: {}'.format(i[2]))
            # Read data
            org = IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype("float32")
            org = org[np.newaxis, :]#(ch, z, y, x)
            mask = IO.read_mhd_and_raw(os.path.join(self._root, i[2]))
            # Extract mask coordinate
            mask_co = np.where(mask == 1) # shape([xyz][coordinate index])
            # Change to coordinate  
            label_ = IO.read_mhd_and_raw(os.path.join(self._root, i[1])).flatten()
            label = np.zeros((org.shape[1]*org.shape[2]*org.shape[3], self._max_label), dtype=int)
            # one-hot encoding
            #"https://stackoverflow.com/questions/29831489/numpy-1-hot-array"
            label[np.arange(org.shape[1]*org.shape[2]*org.shape[3]), label_] = 1
            label = label.transpose().reshape(self._max_label, org.shape[1], org.shape[2], org.shape[3])
            self._dataset.append((org, label, mask_co))

        print(' Initilazation done ')


# class UnetDataset(chainer.dataset.DatasetMixin):
#     def __init__(self, root, data_list_txt, patch_side, number_of_label, patch_size):
#         print(' Initilaze dataset ')
#         self._root = root
#         self._patch_side = patch_side
#         self._max_label = number_of_label #[0, number_of_label)

#         assert(self._patch_side%2==0)

#         """
#         * Read path to org and label data
#         hogehoge.txt
#         org.mhd org_label.mhd
#         """
#         path_pairs = []
#         with open(data_list_txt) as paths_file:
#             for line in paths_file:
#                 line = line.split()
#                 if not line : continue
#                 path_pairs.append(line[:])

#         self._num_of_case = len(path_pairs)
#         print('    # of cases: {}'.format(self._num_of_case))

#         self._dataset=[]
#         for i in path_pairs:
#             print('   Org   from: {}'.format(i[0]))
#             print('   label from: {}'.format(i[1]))
#             print('   liver mask from: {}'.format(i[2]))
#             # Read data
#             org = IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype("float32")
#             mask = IO.read_mhd_and_raw(os.path.join(self._root, i[2]))
#             label_ = IO.read_mhd_and_raw(os.path.join(self._root, i[1])).flatten()
#             # Extract mask coordinate

#             for j in range (patch_size):
#                 mask_co = np.where(mask == 1) # shape([xyz][coordinate index])
#                 org = org[np.newaxis, :]#(ch, z, y, x)
#                 # Change to coordinate  
#                 label = np.zeros((org.shape[1]*org.shape[2]*org.shape[3], self._max_label), dtype=int)
#                 # one-hot encoding
#                 #"https://stackoverflow.com/questions/29831489/numpy-1-hot-array"
#                 label[np.arange(org.shape[1]*org.shape[2]*org.shape[3]), label_] = 1
#                 label = label.transpose().reshape(self._max_label, org.shape[1], org.shape[2], org.shape[3])
#                 self._dataset.append((org, label, mask_co))

#         print(' Initilazation done ')

    def __len__(self):
        if self._validation is None:
            return (int)(self._dataset[0][2][0].size)
        else:
            return (int)(self._validation)

    def get_example(self, i):
        '''
        return (label, org)
        self._dataset[i][2] : mask tuple
        '''
        _, d, h, w = self._dataset[0][0].shape
        number_of_coordinate = self._dataset[0][2][0].size
        candidate = self._dataset[0][2]
        while(1):
            s = np.random.randint(0, number_of_coordinate-1)
            x_s = int (candidate[2][s] - self._patch_side/2)
            x_e = int (x_s +  self._patch_side)
            y_s = int (candidate[1][s] - self._patch_side/2)
            y_e = int (y_s +  self._patch_side)
            z_s = int (candidate[0][s] - self._patch_side/2)
            z_e = int (z_s +  self._patch_side)
            if (z_s>=0 and z_e<d and x_s>=0 and x_e<w and y_s>=0 and y_e < h):
                 break
        
        return self._dataset[0][1][:, z_s:z_e, y_s:y_e, x_s:x_e], self._dataset[0][0][:, z_s:z_e, y_s:y_e, x_s:x_e]
