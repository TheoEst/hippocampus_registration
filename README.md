# hippocampus_registration
Deep learning based registration for Learn2Reg Challenge (Task 4 : MRI Hippocampus)


This repository is a Pytorch implementation  of the participation for the Task 4 of the Learning2Reg Challenge : https://learn2reg.grand-challenge.org/ .

You can also consult the repository for the Task 3 : https://github.com/TheoEst/abdominal_registration


## Use this repository

In order to use this repository, you only need to download the Learn2Reg Task 4 Data : https://learn2reg.grand-challenge.org/Datasets/ and put it on the ./data/ folder. 
You also need to run the preprocessing step to remove the constant padding with the file *zeros_padding.py*.

If you want to apply to run the *pretrain.py* file, you need to download the Oasis 3 dataset (https://www.oasis-brains.org/) and put it on the ./data/ folder.

## Methodology 

Our method comes from the article  *Deep Learning-Based Concurrent Brain Registration and Tumor Segmentation*, **Estienne T., Lerousseau M. et al.**, 2020 (https://www.frontiersin.org/articles/10.3389/fncom.2020.00017/full).

We proposed a 3D Unet for registration trained with 3 losses :
* Reconstruction loss ( Mean Square Error or Local Cross Correlation)
* Segmentation loss ( Dice Loss between deformed segmentation and ground truth segmentation)
* Regularisation loss (To prevent from unrealistic grid)

In the proposed architecture, the moving and fixed image are passed independently through the encoder, and then merged with subtraction operation.

<p align="center">
<img src="https://github.com/TheoEst/hippocampus_registration/blob/main/method.PNG" width="750">
</p>
  
## Models

4 pretrained models are available on the ./models folder : 
* Baseline model
* Baseline model with symmetric training 
* Baseline model with pretraining model (with Oasis 3 dataset)
* Baseline model with pretraining model, trained with both training and validation dataset (used for the test submission)


To recreate this models, launch the following commands :

``` 
python3 -m ./hippocampus_registration.main --crop-size 64 64 64 --zeros-init --batch-size=8 --epochs=600 --session-name=Baseline --lr=1e-4 --instance-norm --data-augmentation --regu-deformable-loss-weight=1e-1 --workers=4 --local-cross-correlation-loss --mse-loss-weight=0 --classic-vnet --plot-mask --deformed-mask-loss --affine-transform --channel-multiplication 8 --deep-supervision 

python3 -m ./hippocampus_registration.main --crop-size 64 64 64 --zeros-init --batch-size=8 --epochs=600 --session-name=Baseline+symmetric --lr=1e-4 --instance-norm --data-augmentation --regu-deformable-loss-weight=1e-1 --workers=4 --local-cross-correlation-loss --mse-loss-weight=0 --classic-vnet --plot-mask --deformed-mask-loss --affine-transform --channel-multiplication 8 --deep-supervision --symmetric-training

python3 -m ./hippocampus_registration.main --crop-size 64 64 64 --zeros-init --batch-size=8 --epochs=600 --session-name=Baseline+symmetric+pretrain --lr=1e-4 --instance-norm --data-augmentation --regu-deformable-loss-weight=1e-1 --workers=4 --local-cross-correlation-loss --mse-loss-weight=0 --classic-vnet --plot-mask --deformed-mask-loss --affine-transform --channel-multiplication 8 --deep-supervision --symmetric-training --model-abspath ./hippocampus_registration/save/models/Pretrain_oasis.pth.tar

python3 -m ./hippocampus_registration.main --crop-size 64 64 64 --zeros-init --batch-size=8 --epochs=600 --session-name=Baseline+symmetric+pretrain --lr=1e-4 --instance-norm --data-augmentation --regu-deformable-loss-weight=1e-1 --workers=4 --local-cross-correlation-loss --mse-loss-weight=0 --classic-vnet --plot-mask --deformed-mask-loss --affine-transform --channel-multiplication 8 --deep-supervision --symmetric-training --merge-train-val --model-abspath ./hippocampus_registration/save/models/Pretrain_oasis.pth.tar  

```

## Prediction

To predict, use the *predict_reg.py* file. 

```
Options : 
  --val                 Do the inference for the validation dataset
  --train               Do the inference for the train dataset
  --test                Replace the validation dataset by test set. (--val is necessary)
  --save-submission     Save the submission in the format for the Learn2Reg challenge
  --save-deformed-img   Save the deformed image and deformed mask in numpy format
  --save-grid           Save the grid in numpy format

Examples :
  python3 -m ./hippocampus_registration.predict_reg  --crop-size 64 64 64 --batch-size=1 --instance-norm  --workers=4 --arch=FrontiersNet --channel-multiplication=8 --classic-vnet  --val --all-dataset --save-submission --model-abspath ./hippocampus_registration/save/models/Baseline+symmetric+pretrain.pth.tar
  
  The prediction will be stored in the folder ./save/submission/Baseline+symmetric+pretrain/ and ./save/pred/Baseline+symmetric+pretrain/
```

## Create submission & evaluation 

To transform the predicted data into a compressed file, just use the *create_submission.py* file. For instance ```python3 ./submission/create_submission.py ./save/submission/Baseline+symmetric+pretrain ```. You will obtain a folder called  *Baseline+symmetric+pretrain_compressed* and a zip file *Baseline+symmetric+pretrain_submission* which you can submit. 

To evaluate the perfomrance, just apply the *apply_evaluation.py* file. For instance ```python3 ./submission/apply_evaluation.py Baseline+symmetric+pretrain_compressed```will generate a csv file in the *./save/evaluation/* folder with all the metrics for each pairs (Dice, Dice30, Hausdorff and standard deviation of Jacobian).

## Performances 

Results on the validation set 

  
Method | Dice | Dice 30 | Hausdorff Distance | Jacobian
------------ | ------------- | ------------ | ------------- | -------------
Baseline  | 0.796 | 0.777 | 2.12 | **0.017**
Baseline + sym.  | 0.830 | 0.818 | 1.68 | 0.018
Baseline + sym. + pretrain | **0.839** |**0.827**  | **1.63** | 0.024 


Example of the results on the validation set :

<p align="center">
<img src="https://github.com/TheoEst/hippocampus_registration/blob/main/results.png" width="500">
</p>


## Developer

This package was developed by Théo Estienne<sup>12</sup>


<sup>1</sup> Université Paris-Saclay, **CentraleSupélec**, *Mathématiques et Informatique pour la Complexité et les Systèmes*, 91190, Gif-sur-Yvette, France.

<sup>2</sup> Université Paris-Saclay, **Institut Gustave Roussy**, Inserm, *Radiothérapie Moléculaire et Innovation Thérapeutique*, 94800, Villejuif, France.
