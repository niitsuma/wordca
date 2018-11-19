
# Author: Hirotaka Niitsuma
# @2018 Hirotaka Niirtsuma
#
# You can use this code olny for self evaluation.
# Cannot use this code for commercial and academical use.
# 
# pantent pending
#  https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
#  Japan patent office patent number 2017-007741

import numpy as np
import scipy
from scipy.sparse import dok_matrix,lil_matrix, csr_matrix,coo_matrix

from delayedsparse import CA

def index2word_load_vocab_freq_rows_txt(filename):
    
    ## load GloVe vocab file
    
    with open(filename, 'r') as f:
        index2word = [x.rstrip().split(' ')[0] for x in f.readlines()]
    return index2word

def load_sparse_coo_bin(filepath,shape,dtype_read=np.float64,dtype_subst=np.uint64,index_shift=1):

    ## load GloVe coo file to coo_matrix
    
    if dtype_subst is None:
        dtype_subst=dtype_read
    row  = []
    col  = []
    data = []
    with open(filepath,"rb") as rfp:
        while True:
            ij = np.fromfile(rfp,np.int32 , 2)
            if ij.size < 2:
                break
            c = np.fromfile(rfp,dtype_read , 1)
            row.append(ij[0]-1+index_shift)
            col.append(ij[1]-1+index_shift)
            data.append(c[0])
    m=coo_matrix((data,(row,col)),shape=shape)
    return m

def index2word_load_csv(filename):
    import csv
    with open(filename+'.index.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        ret=list(spamreader)
        index2word=ret[0]
        return index2word

def index2word_save_csv(filename,index2word):
    import csv
    with open(filename+'.index.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(index2word)

class WordCA:
    def __init__(self,corpus_file_name=None, size=1000, window=30, min_count=5,index_shift=0,vec_mode='F',similarity_mode='dot'):
        self.size=size
        self.correspondenceanalysis=CA(size)
        self.similarity_mode=similarity_mode
        self.vec_mode=vec_mode
        self.index_shift=index_shift
        
        if corpus_file_name is None:
            return
        filename_base=corpus_file_name+'-'+str(min_count)
        fname_index2word=filename_base+'.index.csv'
        
        filename_base2=filename_base+'-'+str(window)
        fname_contingencytable=filename_base2+'.ct.npz'
        fname_correspondenceanalysis=filename_base2+'-'+str(size)+'.dca'
        import os
        if os.path.exists(fname_correspondenceanalysis+'.npz'):
            print('load',fname_correspondenceanalysis)
            self.correspondenceanalysis.load(fname_correspondenceanalysis+'.npz')
            self.index2word=index2word_load_csv(filename_base)
            return
        if os.path.exists(fname_contingencytable):
            print('load',fname_contingencytable)
            self.contingencytable=scipy.sparse.load_npz(fname_contingencytable)
            self.correspondenceanalysis.fit(self.contingencytable)
            self.correspondenceanalysis.save(fname_correspondenceanalysis)
            return
        self.load_concurrence_bin(filename_base,window=window,index_shift=index_shift)
        index2word_save_csv(filename_base,self.index2word)
        scipy.sparse.save_npz(fname_contingencytable, self.contingencytable)
        self.correspondenceanalysis.fit(self.contingencytable)
        self.correspondenceanalysis.save(fname_correspondenceanalysis)
        return
        
    def load_coo_bin(self,filename):
        self.contingencytable = load_sparse_coo_bin(filename,self.shape,np.float64,np.uint64,index_shift=self.index_shift)
        
    def load_concurrence_bin(self,filename_base,window=0,index_shift=0):
        import glob
        import re
        self.index_shift=index_shift
        rest_str='.cooccurrence.bin'
        files=glob.glob(filename_base+'-*'+rest_str)
        assert len(files)> 0
        i_wins= [int(re.findall(r'\d+', f)[-1]) for f in files]
        if window==0:
            self.window=max(i_wins)
        else:
            self.window=window
        self.index2word=index2word_load_vocab_freq_rows_txt(filename_base+'.vocab.txt')
        self.n_vocab=len(self.index2word)
        self.shape=(self.n_vocab+self.index_shift, self.n_vocab+self.index_shift)
        fn=filename_base+'-'+str(self.window)+rest_str
        self.load_coo_bin(fn)

        

def word_similarity_model(model,x, y):
    if not hasattr( model, "vec_mode" ): 
        model.vec_mode='F'
        print('set model.vec_mode=F in word_similarity_model')
    if x not in model.index2word:
        return 0
    elif y not in model.index2word:
        return 0
    else:
        if hasattr( model, "index_shift" ):
            ix=model.index2word.index(x)+model.index_shift
            iy=model.index2word.index(y)+model.index_shift
        else:
            ix=model.index2word.index(x)
            iy=model.index2word.index(y)
        xvec=vars(model.correspondenceanalysis)[model.vec_mode][ix]
        yvec=vars(model.correspondenceanalysis)[model.vec_mode][iy]
        if hasattr( model, "similarity_mode" ):
            if model.similarity_mode=='ndot':
                den=np.linalg.norm(xvec)*np.linalg.norm(yvec)
                if den == 0.0:
                    return 0
                else:
                    return xvec.reshape((1,-1)).dot(yvec.reshape((-1,1)))[0][0]/den

        return xvec.reshape((1,-1)).dot(yvec.reshape((-1,1)))[0][0]


def eval_ws(model,wsfile='testsets/ws/ws353_similarity.txt'):
    def read_ws_test_set(path):
        test = []
        with open(path) as f:
            for line in f:
                x, y, sim = line.strip().lower().split()
                test.append(((x, y), sim))
        return test
    from scipy.stats.stats import spearmanr
   
    data=read_ws_test_set(wsfile)                
    results = []
    for (x, y), sim in data:
        results.append((word_similarity_model(wca,x, y), sim))
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0]

if __name__ == '__main__':
    import sys
    if len(sys.argv)==2:
        corpus=sys.argv[1]
        print(corpus)
        #corpus='text01'
        wca=WordCA(corpus)
        wsfile='testsets/ws/ws353_similarity.txt'
        print(wsfile)
        print('word similarity score=',eval_ws(wca))



