from sklearn import metrics
#from Model.setup_data import *
import pickle
import tensorflow as tf
# Source for DBN implementation:  https://github.com/albertbup/deep-belief-network
#from Model.dbn.tensorflow import SupervisedDBNClassification

def getCNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN")
    elif (feature_type == 0) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/With IP/CNN')
    elif (feature_type == 1) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN')
    elif (feature_type == 1) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN')

    elif (feature_type == 2) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN')
    elif (feature_type == 2) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN')

    elif (feature_type == 3) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN')
    elif (feature_type == 3) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN')
    
    elif (feature_type == 4) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN')
    elif (feature_type == 4) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN')
    
    elif (feature_type == 5) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN')
    elif (feature_type == 5) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN')

def getDNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return tf.keras.models.load_model("./Resources/models/Binary/With IP/DNN")

    elif (feature_type == 0) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/With IP/DNN')
    elif (feature_type == 1) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/DNN')
    elif (feature_type == 1) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/Binary/Without IP/DNN')

    elif (feature_type == 2) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/DNN')
    elif (feature_type == 2) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/DNN')

    elif (feature_type == 3) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/DNN')
    elif (feature_type == 3) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/DNN')
    
    elif (feature_type == 4) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/DNN')
    elif (feature_type == 4) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/DNN')
    
    elif (feature_type == 5) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/DNN')
    elif (feature_type == 5) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/DNN')

    

def getRNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return tf.keras.models.load_model("./Resources/models/Binary/With IP/Simple RNN")

    elif (feature_type == 0) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/With IP/Simple RNN')
    elif (feature_type == 1) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/Simple RNN')
    elif (feature_type == 1) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/Binary/Without IP/Simple RNN')

    elif (feature_type == 2) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/Simple RNN')
    elif (feature_type == 2) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/Simple RNN')

    elif (feature_type == 3) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/Simple RNN')
    elif (feature_type == 3) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/Simple RNN')
    
    elif (feature_type == 4) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/Simple RNN')
    elif (feature_type == 4) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/Simple RNN')
    
    elif (feature_type == 5) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/Simple RNN')
    elif (feature_type == 5) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/Simple RNN')

def getLSTM(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return tf.keras.models.load_model("./Resources/models/Binary/With IP/LSTM-Flatten")

    elif (feature_type == 0) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/With IP/LSTM-Flatten')
    elif (feature_type == 1) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/LSTM-Flatten')
    elif (feature_type == 1) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/Binary/Without IP/LSTM-Flatten')

    elif (feature_type == 2) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM')
    elif (feature_type == 2) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM')

    elif (feature_type == 3) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM')
    elif (feature_type == 3) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM')
    
    elif (feature_type == 4) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM')
    elif (feature_type == 4) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM')
    
    elif (feature_type == 5) & (classification_type == 0):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM')
    elif (feature_type == 5) & (classification_type == 1):
        return tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM')

def getMLP(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return pickle.load(open("./Resources/models/Binary/With IP/MLP/MLP.sav", 'rb'))

    elif (feature_type == 0) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/With IP/MLP/MLP.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/Without IP/MLP/MLP.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 0):
        return pickle.load(open('./Resources/models/Binary/Without IP/MLP/MLP.sav', 'rb'))

    elif (feature_type == 2) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/MLP/MLP.sav', 'rb'))
    elif (feature_type == 2) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/MLP/MLP.sav', 'rb'))

    elif (feature_type == 3) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/MLP/MLP.sav', 'rb'))
    elif (feature_type == 3) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/MLP/MLP.sav', 'rb'))
    
    elif (feature_type == 4) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Binary/MLP/MLP.sav', 'rb'))
    elif (feature_type == 4) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Multi-class/MLP/MLP.sav', 'rb'))
    
    elif (feature_type == 5) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Binary/MLP/MLP.sav', 'rb'))
    elif (feature_type == 5) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/MLP/MLP.sav', 'rb'))

