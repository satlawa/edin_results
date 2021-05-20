# loading dataset

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, sampler


class ForestDataset(torch.utils.data.Dataset):

    '''Characterizes a dataset for PyTorch'''

    def __init__(self, path, ground_truth='ground_truth_std'):
        '''Initialization'''
        # open dataset
        self.dset = h5py.File(path, 'r')
        self.prediction = self.dset['prediction']
        self.ground_truth = self.dset[ground_truth]

        # set number of samples
        self.dataset_size = self.ground_truth.shape[0]


    def __len__(self):
        '''Denotes the total number of samples'''
        return self.dataset_size


    def __getitem__(self, index):
        '''Generates one sample of data'''
        
        y_hat = torch.tensor(self.prediction[index][:,:,0], dtype=torch.torch.uint8)
        y = torch.tensor(self.ground_truth[index][:,:,0], dtype=torch.torch.uint8)

        return y_hat, y #torch.from_numpy(y).permute(2, 0, 1)


    def close(self):
        ''' closes the hdf5 file'''
        self.dset.close()


    def get_sampler(self, split={'train':0.8, 'val':0.1, 'test':0.1}, \
                    shuffle_dataset=True, random_seed=0, chunk_size=0, fold=0):
        # create indices
        if chunk_size > 0:
            # case tiles 256x256
            dsize = int(self.dataset_size/4)
            indices = (np.arange(self.dataset_size/4)).astype(int)
        else:
            # case tiles 512x512
            dsize = self.dataset_size
            indices = np.arange(self.dataset_size)
            
        # shuffle dataset
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        # split test
        split_test = int(np.floor(split['test'] * dsize))
        test_indices = indices[:split_test]
        
        # split validation
        split_val = int(np.floor(split['val'] * dsize))
        # calc fold for crossvalidation
        val_start = split_test + split_val*fold
        val_end = split_test + split_val*(fold+1)
        val_indices = indices[val_start:val_end]
        
        # split train
        # calc fold for crossvalidation
        if sum(split.values()) == 1.0:
            if val_start == split_test:
                train_indices = indices[val_end:]
            else:
                train_indices = np.concatenate((indices[split_test:val_start], indices[val_end:]))
        else:
            if val_start == split_test:
                train_indices = indices[val_end:split_val+split_test+split_train]
            else:
                train_indices = np.concatenate((indices[split_test:val_start], \
                                                indices[val_end:split_val+split_test+split_train]))
            
        if chunk_size > 0:
            train_indices = self.extend_idxs(train_indices, chunk_size, 4, dsize)
            val_indices = self.extend_idxs(val_indices, chunk_size, 4, dsize)
            test_indices = self.extend_idxs(test_indices, chunk_size, 4, dsize)
            
        # set sampler
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        val_sampler = sampler.SubsetRandomSampler(val_indices)
        test_sampler = sampler.SubsetRandomSampler(test_indices)

        return train_sampler, val_sampler, test_sampler
        

    def extend_idxs(self, idxs, chunk_size, ext, dsize):
        '''
        create indices of dataset that is smaller (faster learing e.g. 256) from a initially 
        bigger dataset (e.g. 512) so that the train, validation and test splits are the same.

        input:
            idxs (np.array) = indices
            chunk_size (int) = chunk_size
            ext (int) = extend in how many pieces was the initial image cutted
        '''

        # create empty array to store indices
        idxs_ext = np.zeros(idxs.shape[0]*ext)
        #
        lim = dsize//chunk_size*chunk_size
        chunk_size_last = dsize - lim
        # convert original indices to extended indices
        idxs_trans = np.zeros((idxs.shape[0]))
        #idxs_trans = ((idxs//chunk_size)*(chunk_size*ext) + (idxs%chunk_size)).astype(int)
        idxs_trans[idxs>=lim] = ((idxs[idxs>=lim]//chunk_size)*(chunk_size_last*ext) + \
                                 (idxs[idxs>=lim]%chunk_size_last)).astype(int)
        idxs_trans[idxs<lim] = ((idxs[idxs<lim]//chunk_size)*(chunk_size*ext) + \
                                (idxs[idxs<lim]%chunk_size)).astype(int)
        # fill in data
        for i in range(ext):
            idxs_temp = np.zeros((idxs.shape[0]))
            idxs_temp[idxs_trans>=lim*ext] = idxs_trans[idxs_trans>=lim*ext] + (chunk_size_last*i)
            idxs_temp[idxs_trans<lim*ext] = idxs_trans[idxs_trans<lim*ext] + (chunk_size*i)

            idxs_ext[idxs.shape[0]*i:idxs.shape[0]*(i+1)] = idxs_temp

        return idxs_ext.astype(int) 


    def show_item(self, index):
        '''shows the data'''
        #plt.imshow(np.array(self.ground_truth[index]))

        fig = plt.figure(figsize=(20,20))

        dic_data = {'Prediction' : [np.array(self.prediction[index].astype('f')), [0, 1, 2, 3, 4]],
        'Ground Truth' : [np.array(self.ground_truth[index].astype('f')), [0, 1, 2, 3, 4]]}

        for i, key in enumerate(dic_data):
            ax = fig.add_subplot(1, 2, i+1)
            imgplot = plt.imshow(dic_data[key][0])
            ax.set_title(key)
            plt.colorbar(ticks=dic_data[key][1], orientation='horizontal')
            plt.axis('off')
