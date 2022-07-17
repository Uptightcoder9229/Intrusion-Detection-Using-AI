
import pickle
import tensorflow as tf


def getCNN_shallow(feature_type, classification_type):
    model = {}
    if (feature_type == 0) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/With IP/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/With IP/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/With IP/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/With IP/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 0) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/With IP/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/With IP/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/With IP/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/With IP/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/Without IP/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/Without IP/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/Without IP/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/Without IP/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/Without IP/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/Without IP/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/Without IP/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/Without IP/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/Without IP/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/Without IP/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 3) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 4) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 4) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 5) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 5) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-Shallow Model/RF-model.sav", 'rb')) 
        return model


def getRNN_shallow(feature_type, classification_type):
    model = {}
    if (feature_type == 0) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/With IP/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/With IP/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/With IP/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/With IP/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 0) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/With IP/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/With IP/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/With IP/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/With IP/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/Without IP/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/Without IP/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/Without IP/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/Without IP/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/Without IP/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/Without IP/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/Without IP/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/Without IP/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/Without IP/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/Without IP/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 3) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 4) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 4) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 5) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 5) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RNN-RNN-Shallow Model/RF-model.sav", 'rb')) 
        return model


def getLSTM_shallow(feature_type, classification_type):
    model = {}
    if (feature_type == 0) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/With IP/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/With IP/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/With IP/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/With IP/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 0) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/With IP/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/With IP/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/With IP/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/With IP/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/Without IP/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Multi-class/Without IP/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Multi-class/Without IP/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Multi-class/Without IP/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Multi-class/Without IP/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/Without IP/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/Binary/Without IP/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/Binary/Without IP/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/Binary/Without IP/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/Binary/Without IP/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 2) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Binary/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model

    elif (feature_type == 3) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 4) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Binary/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 4) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithIP Top11/Multi-class/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    
    elif (feature_type == 5) & (classification_type == 0):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Binary/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
    elif (feature_type == 5) & (classification_type == 1):
        model['main'] = tf.keras.models.load_model("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM-Shallow Model")
        model['KNN'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM-Shallow Model/KNN-model.sav", 'rb'))
        model['SVM'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM-Shallow Model/SVM-model.sav", 'rb'))
        model['DT'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM-Shallow Model/DT-model.sav", 'rb'))
        model['RF'] = pickle.load(open("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/LSTM-LSTM-Shallow Model/RF-model.sav", 'rb')) 
        return model
