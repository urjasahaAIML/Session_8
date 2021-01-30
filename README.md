## Session_8


# Assignment:

Train and test with resnet18 model. Code has to be in modularized.

# Result

88.30% validation accuracy reached in 40 epochs




# modules:
1. main.py:             Controller module. It can orchestrate the workflow to load data, train and test a model
2. dataloader.py:       This is the data loader module. 
3. model_resnet.py:     Defines Resnet18 model
4. utils.py:            Utilities to display sample images etc
5. hyperparameters.py:  Defines learning rate, batch size, scheduler milestones etc
6. training.py:         Defines function for creating a model, print model summary, test the model
7. testing.py:          Defines function testing a trained model, display statts about hits and misses in prediction.

# Model
Resnet18 

# Optimizer
scheduler used (please refer to training.py)

# Regularization:
Batch Norm (please refer to model_resnet.py)

# Main Notebook link:


# Functional Modules:

main.py: 			https://github.com/tapasML/Session_8/blob/main/main.py

dataloader.py: 		https://github.com/tapasML/Session_8/blob/main/dataloader.py

training.py:		https://github.com/tapasML/Session_8/blob/main/training.py

testing.py:			https://github.com/tapasML/Session_8/blob/main/testing.py

utils.py:			https://github.com/tapasML/Session_8/blob/main/utils.py

model_resnet.py:	https://github.com/tapasML/Session_8/blob/main/model_resnet.py

hyperparameters.py:	https://github.com/tapasML/Session_8/blob/main/hyperparameters.py



