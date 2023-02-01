# Position-Guided Transformer for Image Captioning
This repository contains the reference code for the paper _[Position-Guided Transformer for Image Captioning]

<p align="center">
  <img src="images/PGT.png" alt="Position-Guided Transformer" width="960"/>
</p>

## Environment setup
Clone the repository and create the `m2release` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2release
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.6 and PyTorch (>1.8.0) is required to run our code. 


## Data preparation
To run the code, annotations, evaluation tools and visual features for the COCO dataset are needed.  

Firstly, most annotations have been prepared by [1], please download [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and rename the extracted folder as annotations, please download [image_info_test2014.json](http://images.cocodataset.org/annotations/image_info_test2014.zip) and put it into annotations. 

Secondly, please download the [evaluation tools](https://pan.baidu.com/s/1vP7Mt1gLLvn4HNxxOvSYxg) (Access code: xh6e) and extarct it in the project root directory.

Then, visual features are computed with the code provided by [2]. To reproduce our result, please download the COCO features file in [ResNeXt_101/trainval](https://pan.baidu.com/s/1s4B7JCrIk7CrQoFx5WOgjQ) (Access code:bnvu) and extract it as X101_grid_feats_coco_trainval.hdf5.


## Evaluation
To reproduce the results reported in our paper, download the pretrained model file [DSNT.pth](https://pan.baidu.com/s/164dndxWvI1FN7kNtftmzSA) (Access code:gvnn) and place it in the code folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 40) |
| `--workers` | Number of workers (default: 8) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |


## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 40) |
| `--workers` | Number of workers (default: 8) |
| `--head` | Number of heads (default: 4) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our model with the parameters used in our experiments, use
```
python train.py --exp_name PGT --batch_size 40 --head 4 --features_path /path/to/features --annotation_folder /path/to/annotations
```


#### References
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   


#### Acknowledgements
Thanks Cornia _et.al_ [M2 transformer](https://github.com/aimagelab/meshed-memory-transformer),
       Zhang _et.al_ [RSTNet](https://github.com/zhangxuying1004/RSTNet), and
       Luo _et.al_ [DLCT](https://github.com/luo3300612/image-captioning-DLCT) for their open source code.
       
Thanks Jiang _et.al_ for the significant discovery in visual representation [2].
