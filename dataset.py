import numpy as np

class Dataset(object):
    def __init__(self, filepath, mode=None, imsize=None):
        if mode == 'train':
            data = np.load(filepath).items()[0][1][0]
        elif mode == 'test':
            data = np.load(filepath).items()[0][1][1]
        else:
            raise ValueError('mode can be either train or test.')
            
        self._num_examples = data.shape[0]
        self._labels = data[:, 0]
        self._s1 = data[:, 1]
        self._s2 = data[:, 2]
        self._images = data[:, 3:]
        if imsize is not None: # For Convolutions
            self._images = self._images.reshape([self._num_examples, imsize, imsize, -1])
        
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def images(self):
        return self._images
    
    @property
    def s1(self):
        return self._s1
    
    @property
    def s2(self):
        return self._s2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """Return next 'batch_size' examples from this data set.
        """
        # Check: batch size should not exceed the size of dataset
        assert batch_size <= self._num_examples
        
        # Initial index for slicing
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        # Not enough data for a batch: Reset + Shuffling
        if self._index_in_epoch > self._num_examples:
            # Increment finished epoch
            self._epochs_completed += 1
            # Shuffule the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._s1 = self._s1[perm]
            self._s2 = self._s2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
           
        # End index for slicing
        end = self._index_in_epoch
        
        return self._images[start:end], self._s1[start:end], self._s2[start:end], self._labels[start:end]
