
# Deepfake Detection using Deeplearning

This project is focused on detecting deepfake videos using deep learning techniques. Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. The objective is to create a model that can accurately distinguish between real and manipulated videos.

# Table of contents                      

  * [Introduction](#Introduction)
  * [Dataset](#Dataset)
  * [Model Architecture](#Model-Architecture)
  * [Installation](#installation)
  * [Usage](#Usage)
  * [Contributors](#Contributors)

# Introduction

Deepfake technology leverages deep learning algorithms to create fake media content that appears genuine. This technology can be misused for spreading false information, creating fake identities, and more. Our project aims to develop a robust model to identify and classify deepfake videos, thereby contributing to efforts in maintaining the authenticity of media content.

# Dataset

The datasets used in this project include:


[FaceForensics++ ](https://github.com/ondyari/FaceForensics) :A dataset containing manipulated videos for research on face forensics.

[DeepFake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge) :A comprehensive dataset provided for the deepfake detection challenge.

# Model Architecture

The model architecture includes:
* Convolutional Neural Networks (CNNs) for feature extraction.
* Pooling Layers for dimensionality reduction.
* Fully Connected Layers for classification.

# Installation

Follow these steps to set up the project locally:

1. Clone the repository:
```bash
git clone https://github.com/BhayanakAatma/Deepfake-detection-using-Deeplearing.git
cd deepfake-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install streamlit 
```bash
pip install streamlit
streamlit hello
```

# Usage

1. Open the file directory in command prompt or terminal.

2. Run the below command
```bash 
python -m streamlit run Output.py
```

3. After that the browser should automatically open by itself, if not copy the link shown in the terminal and paste it in the browser.

http://localhost:8501

## Contributors

We would like to thank all the contributors who have helped in making this project better!

- **[Lucky Nirankari](https://github.com/BhayanakAatma)** - *developed website, API, and optimizing the model for the project*
- **[Mohd. Salim Shaik ](https://github.com/Salim-333)** - *Designing neural network architecture*
- **[Vasant Pardeshi](https://github.com/Vas1261)** - *developed algorithm and documentation*

