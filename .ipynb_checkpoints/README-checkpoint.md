# Multi Word Expression (MWE) classifier with enriched morpho syntactic features 

The architectures variants implemented here are fixed length MLP and RNN with conditional random fields([CRF](https://arxiv.org/abs/1508.01991)). The hyperparameters are tuned using grid search.

### 1. How to train the models :

a. You can either train the model on jupyter notebook with the script provided in the two train notebooks: "train_models.ipynb";

b. Or you can run the following commands from your terminal, supposed that we are in the working directory:

​     To install the relevant dependencies:

```
pip install -r requirements.txt
```

​     To train a MLP model

```
python ml[_classifier.py mlp.yaml
```

​     To train a RNN model:

```
python rnn_classifier.py rnn.yaml
```

​If you want to change other variants than vanilla RNN, you can change the "NAME" argument inside the yaml file to "LSTM" or "ATTRNN".

### 2. The corpus used to train the classifier is the french tree bank, it can be found in the folder "corpus":

We extracted simple mwe tags("B", "I", or "B", "I", "O") combined the pos tags. 

Stats:


```





