## NYU-CV-Fall-2019

### Assignment 2: Traffic sign competition

Continued from : https://github.com/anujmenta/traffic-sign-detection-homework

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Training and validating your model
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `data.py` for your assignment, with an aim to make the validation score better.

- By default the images are loaded and resized to 32x32 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.
- By default a validation set is split for you from the training set and put in `[datadir]/val_images`. See data.py on how this is done.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```
