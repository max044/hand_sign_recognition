# hand_sign_recognition
Hand sign recognition using a specific nearest neighbors algorithm for high dimensionalities.

# Why this project ?
I did this project during my first semester at **SFSU**. It is my term project of Big Data class.<br>
I worked on it during 2 month and the goal was to read, understand and implement a research paper.<br>
The paper I worked on has been write by Sameer A. Nene and Shree K. Nayar is available [here](https://cave.cs.columbia.edu/old/publications/pdfs/Nene_PAMI97.pdf "A Simple Algorithm for Nearest Neighbor Search in High Dimensions").

# How to run
<ol>
  <li>Clone repository</li>
  <li>Download the <a href="https://www.kaggle.com/datasets/datamunge/sign-language-mnist">mnist dataset</a></li>
  <ol>
    <li>Concat train and test csv on a single file (34628 lines)</li>
  </ol>
  <li> Or you can download full dataset directly from <a href="https://drive.google.com/file/d/1CC84O7caoeCVOUCeaXTMjs2r-iuw7jax/view?usp=share_link">here</a></li>
  <li>Add the dataset in "data" folder inside "models_code"</li>
  <li>Create env and install the requirements</li>
  <li>open "process_fnn.ipynb" and run cells</li>
</ol>

## Info
knn_env python version: 3.10 (virtual environment not shared in sources).<br>
Source code is in **models_code** folder.<br>
Research paper and presentation are inside the **documents** folder.
