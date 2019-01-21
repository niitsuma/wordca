
## Word Correspondence Analysis(WCA)
Distributed words epresentations using Correspondence Analysis

more description in  https://arxiv.org/abs/1605.05087


## demo 
```sh
>>> pip3 install delayedsparse numba 
>>> git clone https://github.com/niitsuma/wordca
>>> cd wordca
>>> bash  demo.sh
```

## Usage

```sh
CORPUS=text8
MIN_COUNT=5
WINDOW=24
VECTOR_SIZE=8000
bash tailcut.sh $CORPUS $MIN_COUNT $WINDOW $VECTOR_SIZE
```

### wor2vec format result 
text8-5-24-1-tailcut-8000.F.vec
is the result in word2vec format.
The computed result can be downloaded from
http://www.suri.cs.okayama-u.ac.jp/~niitsuma/wordca/text8-5-24-1-tailcut-8000.F.vec.bz2

### correspondence analysis result
text8-5-24-1-tailcut-8000.dca.npz
contains various information about correspondence analysis.
Plz see save and load function in 
https://github.com/niitsuma/delayedsparse/blob/master/delayedsparse/ca.py 



## License

@2018 Hirotaka Niitsuma.

You can use these codes olny for self evaluation.
Cannot use these codes for commercial and academical use.

* pantent pending
  * https://patentscope2.wipo.int/search/ja/detail.jsf?docId=JP225380312
  * Japan patent office:patent number 2017-007741 , 2018-126430



## Author
Hirotaka Niitsuma.


@2018 Hirotaka Niitsuma.

