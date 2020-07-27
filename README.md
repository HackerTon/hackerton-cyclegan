# Hackerton Implementation of Cyclegan

### NOTES

Currently only support celebsa dataset.

## Instructions:

### Prerequisite

```
Able to run on
python => 3.7.7
tensorflow => 2.2.0
tensorflow-examples => latest version
```

### How to Run

```
Require arguments:
-d : path to the dataset

Optional arguments(default):
--gpu : used to enable gpu mode (no set)
--lmbda : lambda constant (10)
--nsamples : number of samples to be taken from dataset (1000)
--niters : number of iterations (200)
--width : width of images to be trained on (256)

```

Step 1: `python train.py -d ~/dataset`

## TODO

play around with network size
