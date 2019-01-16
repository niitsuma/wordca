
# Author: Hirotaka Niitsuma
# @2018 Hirotaka Niitsuma
#
# You can use this code olny for self evaluation.
# Cannot use this code for commercial and academical use.
# 
# pantent pending
#  https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
#  Japan patent office patent number 2017-007741, 2018-126430

import os
import numpy as np
import scipy
from scipy.sparse import dok_matrix,lil_matrix, csr_matrix,coo_matrix,isspmatrix_coo,isspmatrix_csc,isspmatrix_csr

from delayedsparse import CA

from numba import jit, prange,njit


def index2word_load_vocab_freq_rows_txt(filename):
    
    ## load GloVe vocab file
    
    with open(filename, 'r') as f:
        index2word = [x.rstrip().split(' ')[0] for x in f.readlines()]
    return index2word


# @jit('[int32,int32,float64]()')
# def load_sparse_coo_bin_list(rfp,shape,dtype_read,dtype_subst,index_shift):
#     row  = []
#     col  = []
#     data = []
#     #with open(filepath,"rb") as rfp:
#     while True:
#         ij = np.fromfile(rfp,np.int32 , 2)
#         if ij.size < 2:
#             break
#         c = np.fromfile(rfp,dtype_read , 1)
#         row.append(ij[0]-1+index_shift)
#         col.append(ij[1]-1+index_shift)
#         data.append(c[0])
#     return [row,col,data]

#@jit(nopython=True, parallel=True)
def load_sparse_coo_bin(filepath,shape,dtype_read=np.float64,dtype_subst=np.uint64,index_shift=1):
    ## load GloVe coo file to coo_matrix
    
    if dtype_subst is None:
        dtype_subst=dtype_read
    row  = []
    col  = []
    data = []
    with open(filepath,"rb") as rfp:
        #[row,col,data]=load_sparse_coo_bin_list(rfp,shape,dtype_read,dtype_subst,index_shift)
        while True:
            ij = np.fromfile(rfp,np.int32 , 2)
            if ij.size < 2:
                break
            c = np.fromfile(rfp,dtype_read , 1)
            row.append(ij[0]-1+index_shift)
            col.append(ij[1]-1+index_shift)
            data.append(c[0])
    #print(min(row),min(col))
    #print(max(row),max(col))
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

@jit(nopython=True, parallel=True)
def rate_truncated_contingencytable_coo_sub(data,row,col,ratevec,sum_in_row):
    for l in prange(len(data)):
        i=row[l]
        j=col[l]
        if data[l] < sum_in_row[i]*ratevec[j]:
            data[l] =0
                
def rate_truncated_contingencytable(ct,ratevec,sum_in_row=None):
    if sum_in_row is None:
        sum_in_row=np.array(ct.T.sum(axis=0))[0,:]
    if isspmatrix_coo(ct):
        rate_truncated_contingencytable_coo_sub(ct.data,ct.row,ct.col,ratevec,sum_in_row)
    # elif isspmatrix_csc(ct) or isinstance(ct,csc_file_matrix):
    #     rate_truncated_contingencytable_csc_sub(ct.shape[1],ct.data,ct.indptr,ct.indices,ratevec,sum_in_row)
    # elif isspmatrix_csr(ct) or isinstance(ct,csr_file_matrix):
    #     rate_truncated_contingencytable_csr_sub(ct.shape[0],ct.data,ct.indptr,ct.indices,ratevec,sum_in_row)
    else:
        raise NotImplementedError("not implemented")
    ct.eliminate_zeros()
    return ct

        
class WordCA:
    def __init__(self,corpus_file_name=None, size=8000, window=24, min_count=5,index_shift=1,vec_mode='F',similarity_mode='dot',contingencytable_mode='tailcut'):
        self.size=size
        self.correspondenceanalysis=CA(size)
        self.similarity_mode=similarity_mode
        self.vec_mode=vec_mode
        self.index_shift=index_shift
        
        if corpus_file_name is None:
            return

        filename_base0   =corpus_file_name+'-'+str(min_count) 
        filename_base1   =filename_base0 +'-'+str(window) 
        fname_index2word =filename_base1+'.index.csv'
        filename_base2   =filename_base1 +'-'+str(index_shift)+'-'+contingencytable_mode
        fname_contingencytable      =filename_base2+'.ct.npz'        
        fname_correspondenceanalysis=filename_base2+'-'+str(size)+'.dca'

        if os.path.exists(fname_index2word):
            print('load', fname_index2word)
            self.index2word=index2word_load_csv(filename_base1)
        else:
            print('load', filename_base0+'.vocab.txt')
            self.index2word=index2word_load_vocab_freq_rows_txt(filename_base0+'.vocab.txt')
            index2word_save_csv(filename_base1,self.index2word)
            
        print('len index2word',len(self.index2word))            
        
        if os.path.exists(fname_correspondenceanalysis+'.npz'):
            print('load', fname_correspondenceanalysis)
            self.correspondenceanalysis.load(fname_correspondenceanalysis+'.npz')
            return
        if os.path.exists(fname_contingencytable):
            print('load', fname_contingencytable)
            self.contingencytable=scipy.sparse.load_npz(fname_contingencytable)
        else:
            if contingencytable_mode == 'glove':
                self.contingencytable=self.load_concurrence_bin(filename_base0,window=window,index_shift=index_shift,fname_mode='gloveco')
            elif contingencytable_mode== 'tailcut':
                contingencytable0=self.load_concurrence_bin(filename_base0,window=1,index_shift=index_shift,fname_mode='cooccurrence')
                sc=contingencytable0.sum()
                #contingencytables=[self.load_concurrence_bin(filename_base0,window=k+1,index_shift=index_shift,fname_mode='cooccurrence')  for k in range(window)]
                #sc=contingencytables[0].sum()
                ratevec =np.array(contingencytable0.sum(axis=0))[0, :]
                ratevec /= sc
                #self.contingencytable=0
                self.contingencytable =rate_truncated_contingencytable(contingencytable0,ratevec)
                del contingencytable0
                for k in range(1,window+1):
                    self.contingencytable += rate_truncated_contingencytable(self.load_concurrence_bin(filename_base0,window=k,index_shift=index_shift,fname_mode='cooccurrence'),ratevec)
                    #self.contingencytable += rate_truncated_contingencytable(contingencytables[k],ratevec)
            scipy.sparse.save_npz(fname_contingencytable, self.contingencytable)
        self.correspondenceanalysis.fit(self.contingencytable)
        self.correspondenceanalysis.save(fname_correspondenceanalysis)
        return

        
    def load_concurrence_bin(self,filename_base,window=0,index_shift=0,fname_mode='gloveco'):
        import glob
        import re
        self.index_shift=index_shift
        #rest_str='.cooccurrence.bin'
        rest_str='.'+fname_mode+'.bin'
        files=glob.glob(filename_base+'-*'+rest_str)
        assert len(files)> 0
        i_wins= [int(re.findall(r'\d+', f)[-1]) for f in files]
        if window==0:
            self.window=max(i_wins)
        else:
            self.window=window
        self.n_vocab=len(self.index2word)
        self.shape=(self.n_vocab+self.index_shift, self.n_vocab+self.index_shift)
        print('shape',self.shape)
        fn=filename_base+'-'+str(self.window)+rest_str
        print(fn)
        if fname_mode=='cooccurrence':
            assert self.index_shift==1
            return load_sparse_coo_bin(fn,self.shape,np.float64,np.uint64,index_shift=0)
        else:
            return load_sparse_coo_bin(fn,self.shape,np.float64,np.uint64,index_shift=self.index_shift)
        

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
    if len(sys.argv) > 1:
        print(sys.argv)
        corpus=sys.argv[1]
        min_count=int(sys.argv[2])
        window=int(sys.argv[3])
        size=int(sys.argv[4])
        mode=sys.argv[5]
        wca=WordCA(corpus,size=size,window=window,min_count=min_count,contingencytable_mode=mode)
        print('------')
        print('similarity eval')
        for wsfile in ['ws353.txt','ws353_similarity.txt','ws353_relatedness.txt', 'bruni_men.txt', 'radinsky_mturk.txt', 'SimLex999.txt' ,'luong_rare.txt']:
            print(wsfile)
            print('word similarity score=',eval_ws(wca,'testsets/ws/'+wsfile))



