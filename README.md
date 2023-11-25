# PizzaYOLO : A real time pizza object detector

This project is a complete end-to-end training pipeline for object detection applications. The purpose is to create new models which are both fast to create (going from idea to a working model) and fast to run (fast inference time so that models can be deployed). We demonstrate the effectiveness of this pipeline by detecting *pizzas*. It can easily be extended to any objects with **text prompts**. 

The pipeline is composed of 3 steps : 
1. creating a small dataset from web scraping with Selenium
2. automatic labelling using foundational vision models
3. training a lightweight YOLO object detection model (coming soon)

Notes: 
- Web scraping presents legal and ethical challenges (image rights, platform being abused)
- If you are interested in model performance, you should increase dataset size and correct any labelling mistakes
- Zero shot object detector can't currently be used in real time applications due to memory size and low inference time

## Setup

Tested on Windows with WSL2. GPU is recommended for the automatic labelling.

Clone the repositery.

Create a conda environment:  
`conda create -n pizzayolo ...`  
(Requirements coming)  

For Selenium, install geckodriver in WSL:

```
wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz
tar -xvf geckodriver-v0.33.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/
geckodriver --version
```

Install Firefox: (avoid using snap)

```
sudo add-apt-repository ppa:mozillateam/ppa
sudo apt install firefox
```

## Usage

Make sure to follow the script execution order:
1. `scrape_imgs.py` downloads 30 images from Google images to the **/imgs** folder.
2. `annotate_imgs.py` labels all the images in the **/imgs** folder. It outputs a JSON file and the annotated images in **/annotated**. If the model doesn't find any objects on the image, it won't be saved in the **/annotated** folder. So you should compare both folders to check for any missed detections. If you run the model on CPU, you are more likely to notice these missed detections.

File structure:
```
├── README.md
├── annotate_imgs.py
├── annotated
│   ├── annotated.json
│   ├── pizza picnic_image_1.jpg
│   ├── ...
│   └── pizza_image_9.jpg
├── imgs
│   ├── pizza picnic_image_1.jpg
│   ├── ...
│   └── pizza_image_9.jpg
└── scrape_imgs.py
```

## Examples

