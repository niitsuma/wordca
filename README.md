
## Word Correspondence Analysis(WCA)
Distributed words epresentations using Correspondence Analysis

plz see more description about tail-cut kernel in  https://arxiv.org/abs/1605.05087


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

