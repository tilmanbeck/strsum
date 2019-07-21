## Unsupervised Neural Single-Document Summarization of Reviews via Learning Latent Discourse Structure and its Ranking

Corresponding paper: 

https://arxiv.org/abs/1906.05691  
Masaru Isonuma, Juncihiro Mori and Ichiro Sakata (The University of Tokyo)  
Accepted in ACL 2019 as a long paper  


### Requirements

Tensorflow

### Usage

#### Preprocessing

To preprocess the data, run:


#### Training

To train your model, run:

```
python cli.py --mode train --modeldir /path/to/a/model/directory
```

Parameters and logs are saved in `/path/to/a/model/directory`.  
You can change the hyper parameters as described in `cli.py`.

#### Evaluation

To write out and evaluate the summaries generated by your model, run:

```
python cli.py --mode eval --modeldir model_rougel --refdir /path/to/ref --outdir /path/to/out
```

References and system summaries are respectively saved in `/path/to/ref` and `/path/to/out`.  
You have to set the same hyper parameters as used in training.
