# Denoising-Autoencoder
Image denoising with autoencoder

### Setup env
cd project/directory

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

### Files, Folders
- modeling.py: model definition.
- train_model: preprocess data + train + save.
- test_prediction: predict model on test set.
- examples: contains denoised examples

### Data
Data downloaded from [Kaggle Landscape images](https://www.kaggle.com/arnaud58/landscape-pictures)