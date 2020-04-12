from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    #parser.add_argument('--epochs', type=int, default=50)
    #parser.add_argument('--batch-size', type=int, default=64)
    #parser.add_argument('--learning-rate', type=float, default=0.05)
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    model = KNeighborsClassifier(3)                                                 #1.0
    #model = SVC(kernel="linear", C=0.025)                                           #0.6
    #model = SVC(gamma=2, C=1)                                                       #1.0
    #model = GaussianProcessClassifier(1.0 * RBF(1.0))                               #1.0
    #model = DecisionTreeClassifier(max_depth=5)                                     #0.96
    #model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)    #1.0
    #model = MLPClassifier()                                                         #1.0
    #model = MLPClassifier(alpha=1, max_iter=1000)                                   #1.0
    #model = AdaBoostClassifier()                                                    #0.92
    #model = GaussianNB()                                                            #1.0
    #model = QuadraticDiscriminantAnalysis()                                         #0.92
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))