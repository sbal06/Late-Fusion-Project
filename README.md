# A late fusion approach with Image and Text Data

Tweets can be multimodal, in which the image caption gives a semantic description of the image. Before the popularity of multimodal machine learning, previous works included identifying texts and images separately to determine sentiment. 

 ## 📖 Fusion 
Data fusion involves integrating information from multiple souces to create a unified system. There are three common types of fusion in multimodal machine learning. First, early fusion involves combining the different modalities before the feature extraction process. Intermediate fusion involves extracting high-level representations from each modality and fusing them later inside the network. Intermediate fusion is very popular in multimodal machine learning because of the ability to learn rich correlations between the modal's embeddings. In this project, we highlight the pros and cons of late fusion, where the predictions from each modality are combined to form a single final prediction.

 <p align = "center" >
 <image src = "https://github.com/user-attachments/assets/d2313740-e9b8-452b-a373-5ef56e2c7fd2" width = "800" height = "250">
 </p>



## 💾 Files
- **/models/ResNet50.py** - Contains architecture of ResNet50 image model with image augmentation layers.
- **/models/ResNet50contrastiveLearning** - In addition to the ResNet50 architecture, this file contains an implementation of supervised contrastive learning using the N-pairs loss. The model architecutre is defined using Keras Functional API for flexibility in creation.
- **/models/RoBERTa** - Contains architecture of the RoBERTa-base model for sentiment analysis.
- **/models/latefusion/py** - Contains the methods to implement the Detection Rate Approach and Minimizing Loss Approach.
<br>
More files are soon to be added.

## Future Improvements
1. Currently, we are trying to enhance out models by testing them on larger datasets.
2. We are also actively researching other fusion methods to enhance our model.
