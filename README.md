This repository contains the code for the paper: "End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps". 

Code will be released soon. Stay tuned.


This repo contains the official implementation of our paper:
  
**CVPR 2022**  


# Installation 

### Environment
* Python >= 3.7
* PyTorch == 1.8.0


### Data Preprocessing

* SDD (Trajnet split)

1. Download the Trajnet split data from [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet). Put the data under [data/SDD](data/SDD)

2. Run [script](process_trajnet.py) to process the downloaded "train_trajnet.pkl" and "test_trajnet.pkl":
      ```
      python data/SDD/process_trajnet.py
      ``` 


* SDD(P2T split)
1. Download the P2T split data from [P2T](https://github.com/nachiket92/P2T/tree/main/data/sdd). Put the data under [data/SDD](data/SDD)

2. Run [script](process_p2t.py) to process the downloaded "SDDtrain.mat", "SDDval.mat" and "SDDtest.mat":
      ```
      python data/SDD/process_p2t.py
      ``` 
   

* inD 

1. Download the inD data from [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet). Put the data under [data/SDD](data/IND)

2. Run [script](process_trajnet.py) to process the downloaded "train_longterm.pkl" and "test_longterm.pkl":
      ```
      python data/SDD/process_inD.py
      ``` 
      
### Training 


Training the model for Trajnet:

      ```
      python train.py  --dataset "trajnet"
      ``` 
For SDD(p2t split) or inD, the "trajnet" need to be replaced by "sdd" or "ind".   

### Evaluation   

Evaluating on Trajnet dataset:

      ```
      python eval.py  --dataset "trajnet"
      ``` 
For SDD(p2t split) or inD, the "trajnet" need to be replaced by "sdd" or "ind".   

## Citation

```

```

