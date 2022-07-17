from sklearn import metrics
#from Model.setup_data import *
import pickle
import tensorflow as tf


# Source for DBN implementation:  https://github.com/albertbup/deep-belief-network
#from Model.dbn.tensorflow import SupervisedDBNClassification

def getCNN_LSTM(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-LSTM")
        return model
    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/With IP/CNN-LSTM')
        return model
    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN-LSTM')
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN-LSTM')
        return model
    elif (feature_type == 2) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-LSTM')
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-LSTM')
        return model

    elif (feature_type == 3) & (classification_type == 0):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-LSTM')
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-LSTM')
        return model   

    elif (feature_type == 4) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-LSTM')
        return model   
    elif (feature_type == 4) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-LSTM')
        return model    

    elif (feature_type == 5) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-LSTM')
        return model   
    elif (feature_type == 5) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-LSTM')
        return model

def getCNN_RNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-RNN")
        return model
        
    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/With IP/CNN-RNN')
        return model

    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN-RNN')
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN-RNN')
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-RNN')
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-RNN')
        return model

    elif (feature_type == 3) & (classification_type == 0):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-RNN')
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-RNN')
        return model   

    elif (feature_type == 4) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-RNN')
        return model   
    elif (feature_type == 4) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-RNN')
        return model    

    elif (feature_type == 5) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-RNN')
        return model   
    elif (feature_type == 5) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-RNN')
        return model




def getLSTM_LSTM(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/LSTM-LSTM")
        return model
        
    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/With IP/LSTM-LSTM')
        return model

    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/LSTM-LSTM')
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/LSTM-LSTM')
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM')
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM')
        return model
        
    elif (feature_type == 3) & (classification_type == 0):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM')
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM')
        return model   

    elif (feature_type == 4) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM')
        return model   
    elif (feature_type == 4) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM')
        return model    

    elif (feature_type == 5) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM')
        return model   
    elif (feature_type == 5) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM')
        return model

def getRNN_RNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/RNN-RNN")
        return model
        
    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/With IP/RNN-RNN')
        return model

    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/RNN-RNN')
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/RNN-RNN')
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN')
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN')
        return model
        
    elif (feature_type == 3) & (classification_type == 0):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN')
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN')
        return model   

    elif (feature_type == 4) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN')
        return model   
    elif (feature_type == 4) & (classification_type == 1):
        model= {}
        model['main'] =  tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN')
        return model    

    elif (feature_type == 5) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN')
        return model   
    elif (feature_type == 5) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN')
        return model

def getCNN_MLP(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-MLP/CNN")
        model['second'] = pickle.load(open("./Resources/models/Binary/With IP/CNN-MLP/MLP.sav", 'rb'))
        return model

    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/CNN-MLP/CNN")
        model['second'] = pickle.load(open("./Resources/models/Multi-class/With IP/CNN-MLP/MLP.sav", 'rb'))
        return model
        
    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN-MLP/CNN')
        model['second'] = pickle.load(open("./Resources/models/Multi-class/Without IP/CNN-MLP/MLP.sav", 'rb'))
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN-MLP/CNN')
        model['second'] = pickle.load(open("./Resources/models/Binary/Without IP/CNN-MLP/MLP.sav", 'rb'))
        return model
    elif (feature_type == 2) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-MLP/MLP.sav", 'rb'))
            return model
    elif (feature_type == 2) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-MLP/MLP.sav", 'rb'))
            return model

    elif (feature_type == 3) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-MLP/MLP.sav", 'rb'))
            return model
    elif (feature_type == 3) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-MLP/MLP.sav", 'rb'))
            return model
        
    elif (feature_type == 4) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-MLP/MLP.sav", 'rb'))
            return model
    elif (feature_type == 4) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-MLP/MLP.sav", 'rb'))
            return model
        
    elif (feature_type == 5) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-MLP/MLP.sav", 'rb'))
            return model

    elif (feature_type == 5) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-MLP/CNN')
            model['second'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-MLP/MLP.sav", 'rb'))
            return model