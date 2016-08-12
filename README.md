## Seq2Seq Keras

A general purpose library for training seq2seq models on a parallel corpus. No explicit programming is required, training script will take care of preprocessing the data, compiling the model and then training on the corpus. It's a general purpose library, so it can be used for different NLP tasks which requires seq2seq mapping like Text Summarization, Question Answering system, Chatbots etc. 

## Requirements

* keras

* numpy

* theano/tensorflow

* CUDA and CuDNN (if using GPU)

## Example on Machine Translation

On Machine Translation task (translation from English to Hindi), after ~1000 epochs of training (less training data) it was giving following results:

nepal   external        ministry       
 नेपाली   विदेश    UNK     


ramayana        is      an      extraordinary   epic    poetry  written by      poet    valmiki 
 रामायण  कवि     वाल्मीकि द्वारा   लिखा    गया     संस्कृत    का      एक      अनुपम    


 he      is      the     first   black   lrb     UNK     rrb     president       
  वे       इस      देश      के       प्रथम    UNK     -LRB-   अफ्रीकी  UNK     -RRB-   


 administrative  divisions       
 प्रशासनिक        विभाजन  


## TO-DO

* Current parameters hard coded, add argument parser

* Add model saving method

* Add model loading method



