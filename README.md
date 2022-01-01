# Reimplementation of Neural Supersampling for Real-time Rendering
This is the reimplementation of [Neural Supersampling for Real-time Rendering](https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/) by Facebook Reality Labs.

## Requirements

- Python >= 3.5 (3.6 recommended)
- torchvision
- numpy
- tensorboard
- pandas
- tqdm (Optical for test)
- pillow
- matplotlib
- scikit-image

To install other dependencies, you can use pip with :

```bash
pip install -r requirements.txt
```

## Folder Structure

The folder structure of the project is shown below:

```
NSRR-Reimplementation/
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
│
├── config.json - holds configuration for training
├── parse_config.py - class to handle config file and cli options
│
├── base/ - abstract base classes
|   ├── __init__.py
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/ - anything about data loading goes here
|   ├── __init__.py
│   └── data_loaders.py
│
├── data/ - default directory for storing input data
|   ├── All_Scenes/
|   |   ├── train/
|   |   |   ├── img/
|   |   |   ├── flow/
|   |   |   ├── depth/
|   |   ├── test/
|   |   |   ├── img/
|   |   |   ├── flow/
|   |   |   ├── depth/
|   ├── ...(Data of other scenes must also follow the above format)
│
├── model/ - models, losses, and metrics
│   ├── __init__.py
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for tensorboard and logging output
│
├── trainer/ - trainers
│   ├── __init.py__
│   └── trainer.py
│
├── logger/ - module for tensorboard visualization and logging
│   ├── __init__.py
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│  
└── utils/ - small utility functions
    ├── __init__.py
    ├── util.py
    └── ...
```



## Dataset

## Usage

### Basic Configurations

Config files are in `.json` format:

```json
{
    "name": "NSRR",								// training session name
    "n_gpu": 2,								    // number of GPUs to use for training.
    "arch": {
        "type": "NSRR",							// name of model architecture to train
        "args": {
            "upsample_scale": 2					 // rate of supersampling, must be the same as `downsample` below
        }
    },
    "data_loader": {
        "type": "NSRRDataLoader",				 // selecting data loader
        "args":{
            "data_dir": "data/Scene_1/train",	  // dataset path which includes directories 'depth', 'img', and 'flow'
            "batch_size": 1,					// batch size
            "shuffle": true,
            "validation_split": 0.1,			 // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 4,					// number of cpu processes to be used for data loading
            "img_dirname": "img/",
            "depth_dirname": "depth/",
            "flow_dirname": "flow/",
            "downsample": 2,					// rate of supersampling, must be the same as `upsample_scale` above
	    "num_data": 120, 					    // amount of training data
	    "resize_factor": 3					    // reduce the rendered image by 1/resize_factor as ground truth
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,						// learning rate
            "weight_decay": 0,					 // (optional) weight decay
            "amsgrad": true
        }
    },
    "loss": "nsrr_loss",						// loss
    "metrics": [
        "psnr","ssim"							// list of metrics to evaluate
    ],
    "lr_scheduler": {
        "type": "StepLR",						 // learning rate scheduler
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}

```

### Train

#### Using config files

Modify the configurations in `.json` config files and prepare your data, then run:

```bash
python train.py -c config.json
```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

```bash
python train.py --resume path/to/checkpoint
```

By default, the checkpoint is restored in `./saved/model/`

### Test

You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument. 

```bash
python test.py --resume path/to/checkpoint
```

You have to set the path of the test data in the `test.py`.



