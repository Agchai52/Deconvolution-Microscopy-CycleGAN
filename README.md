# Deconvolution-Microscopy-CycleGAN
This is an implementation of CycleGAN with a Blur Kernel for Deconvolution Microscopy: Optimal Transport Geometry.

## Prerequisites
- Python 3.7
- Pytorch, torch>=0.4.1, torchvision>=0.2.1
- To run the code, please install required packages by the following command
```
pip install -r requirements.txt
```

## Preprocess the dataset
1. Generate dataset
```
python generate_dataset.py --phase train --num_imgs 2000
python generate_dataset.py --phase test --num_imgs 500
```
2. Rename the dataset as "dataset".
3. To generate names of all the train and test data, run the file "readDatasetNames.py" 
```
python readDatasetNames.py
```

## Train the model
```
python main.py --phase train --epoch 100 --gpu 0
```

## Test the model
```
python main.py --phase test --gpu 0
```

## Test real images
To do