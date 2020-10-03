# Handwritten Character Recognition

A Handwritten Character Recognition Model using a CRNN model with a CTC Loss. \
This project is currently in progress.

## Prerequisites
To install prerequisites change into root folder after cloning and run:
```sh
pip install -r requirements.txt
```
## Evaluation
To run the model on an image, and get the best path decoding run:
```sh
python extract_text.py
```
Path of the image can be set in `src/extract_text.py`


## Training
To train the model:
1. Copy all the images to `data/words/`
2. Create a text_data.csv which contains names of the file with the corresponding targets.
The  format is as following:
```sh
name,text
file_name1,text1
file_name2,text2
...
```
3. You can change the architecture of the model in `src/Image2Text/configure.py`