This repo contains the official implementation of our paper: "End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps". 
  Ke Guo, Wenxi Liu, Jia Pan.
  
**CVPR 2022**  
[paper](http://arxiv.org/abs/2203.16910)

# Installation 

### Environment
* Python >= 3.7
* PyTorch == 1.8.0


### Data and pretrained model
Please download the pretrained model and data from onedrive([https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3006612_connect_hku_hk/EXqC6hjGTphKh8TkjrwtByEB3FFZ_dpCu0Rs6N7CTG2gag?e=5q4Knz](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3006612_connect_hku_hk/Ei8gZNibG4lBhtJGGkmAGggB1aKgCXg2sYpxViE7PFqkwQ?e=YbB6c4)). Extract the zip file into the main folder.

### Data Preprocessing

Here is the detail of data preprocessing. You can skip it by using the data from google drive. 

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

1. Obtain the processed inD data from [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet). Put the data under [data/SDD](data/IND)

2. Run [script](process_trajnet.py) to process the downloaded "inD_train.pickle" and "inD_test.pickle":
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
@inproceedings{guo2022end,
  title={End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps},
  author={Ke, Guo and Wenxi, Liu and Jia, Pan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

