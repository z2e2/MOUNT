import numpy as np
import pandas as pd
import pickle

class Data:
    '''
    load data
    '''
    
    def __init__(self, DATA_ROOT, MAXLEN, short_AMINO='ARNDCQEGHILKMFPSTWYV', dataset_type='train', train_val_split=0.8):
        
        self.filename = '{}/encoded_data_{}.pkl'.format(DATA_ROOT, dataset_type)
        self.MAXLEN = MAXLEN
        self.short_AMINO = short_AMINO
        self.train_val_split = train_val_split
        self.dataset_type = dataset_type
        self.amino2int = {}
        self.int2amino = {}
        for i in range(len(short_AMINO)):
            self.amino2int[short_AMINO[i]] = int(i)
            self.int2amino[int(i)] = short_AMINO[i]

        self.ONEHOT_DIM = len(short_AMINO)

    def _get_values(self, seq_list):
        
        seq_array = np.zeros((len(seq_list), self.MAXLEN, self.ONEHOT_DIM))
        length_array = np.zeros(len(seq_list), dtype=int)

        for idx, seq in enumerate(seq_list):
            seq_len = len(seq)
            if seq_len > self.MAXLEN:
                seq = seq[:self.MAXLEN]
                length_array[idx] = MAXLEN
            else:
                length_array[idx] = seq_len
            for sidx, base in enumerate(seq):
                if base not in self.short_AMINO:
                    seq_array[idx, sidx, :] = 1/self.ONEHOT_DIM
                else:
                    seq_array[idx, sidx,  self.amino2int[base]] = 1
        
        return seq_array, length_array
    
    def _train_val_split(self, data):
        data_train = data[:self.split_n]
        data_val = data[self.split_n:]
        return data_train, data_val

    def load(self):
        seq_list, y_go, y_pfam, y_ko = pickle.load(open(self.filename, 'rb'))
        self.split_n = int(len(seq_list) * self.train_val_split)
        
        if self.dataset_type == 'train':
            permutation = list(np.random.permutation(len(seq_list)))
            shuffled_seq_list = [seq_list[i] for i in permutation]
            shuffled_y_go = y_go[permutation]
            shuffled_y_pfam = y_pfam[permutation]
            shuffled_y_ko = y_ko[permutation]

            shuffled_seq_list_train, shuffled_seq_list_val = self._train_val_split(shuffled_seq_list)
            shuffled_y_go_train, shuffled_y_go_val = self._train_val_split(shuffled_y_go)
            shuffled_y_pfam_train, shuffled_y_pfam_val = self._train_val_split(shuffled_y_pfam)
            shuffled_y_ko_train, shuffled_y_ko_val = self._train_val_split(shuffled_y_ko)

            train_seq, train_length = self._get_values(shuffled_seq_list_train)
            val_seq, val_length = self._get_values(shuffled_seq_list_val)
        
            return [train_seq, train_length, shuffled_y_go_train.astype(np.float64), shuffled_y_pfam_train.astype(np.float64), shuffled_y_ko_train.astype(np.float64)], [val_seq, val_length, shuffled_y_go_val.astype(np.float64), shuffled_y_pfam_val.astype(np.float64), shuffled_y_ko_val.astype(np.float64)]
        else:
            test_seq, test_length = self._get_values(seq_list)
            return test_seq, test_length, y_go.astype(np.float64), y_pfam.astype(np.float64), y_ko.astype(np.float64)

