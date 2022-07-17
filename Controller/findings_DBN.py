import Model.dbn.tensorflow as dbn
import tensorflow as tf

def getCNN_DBN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-DBN/CNN")
        #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/Binary/With IP/CNN-DBN/DBN.pkl")
        return model

    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/CNN-DBN/CNN")
        #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/Multi-class/With IP/CNN-DBN/DBN.pkl")
        return model
        
    elif (feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN-DBN/CNN')
        #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/Multi-class/Without IP/CNN-DBN/DBN.pkl")
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN-DBN/CNN')
        #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/Binary/Without IP/CNN-DBN/DBN.pkl")
        return model
    elif (feature_type == 2) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('/Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-DBN/CNN')
            ##model['second'] = dbn.SupervisedDBNClassification.load("/Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 2) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-DBN/CNN')
            ##model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-DBN/DBN.pkl")
            return model

    elif (feature_type == 3) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 3) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-DBN/DBN.pkl")
            return model
        
    elif (feature_type == 4) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 4) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-DBN/DBN.pkl")
            return model
        
    elif (feature_type == 5) & (classification_type == 0):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-DBN/DBN.pkl")
            return model

    elif (feature_type == 5) & (classification_type == 1):
            model= {}
            model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-DBN/CNN')
            #model['second'] = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-DBN/DBN.pkl")
            return model

def getDBN_CNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        #model= {}
        #model['main'] = tf.keras.models.load_model("./Resources/models/Binary/With IP/CNN-DBN/CNN")
        model = dbn.SupervisedDBNClassification.load("./Resources/models/Binary/With IP/CNN-DBN/DBN.pkl")
        return model

    elif (feature_type == 0) & (classification_type == 1):
        #model= {}
        #model['main'] = tf.keras.models.load_model("./Resources/models/Multi-class/With IP/CNN-DBN/CNN")
        model = dbn.SupervisedDBNClassification.load("./Resources/models/Multi-class/With IP/CNN-DBN/DBN.pkl")
        return model
        
    elif (feature_type == 1) & (classification_type == 1):
        #model= {}
        #model['main'] = tf.keras.models.load_model('./Resources/models/Multi-class/Without IP/CNN-DBN/CNN')
        model = dbn.SupervisedDBNClassification.load("./Resources/models/Multi-class/Without IP/CNN-DBN/DBN.pkl")
        return model
    elif (feature_type == 1) & (classification_type == 0):
       # model= {}
        #model['main'] = tf.keras.models.load_model('./Resources/models/Binary/Without IP/CNN-DBN/CNN')
        model = dbn.SupervisedDBNClassification.load("./Resources/models/Binary/Without IP/CNN-DBN/DBN.pkl")
        return model
    elif (feature_type == 2) & (classification_type == 0):
            #model= {}
           # model['main'] = tf.keras.models.load_model('/Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("/Resources/models/CFS feature selection/With IP_ Top 7/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 2) & (classification_type == 1):
            #model= {}
            #model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/CNN-DBN/DBN.pkl")
            return model

    elif (feature_type == 3) & (classification_type == 0):
           # model= {}
            #model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/Without IP_Top 6/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 3) & (classification_type == 1):
            #model= {}
           # model['main'] = tf.keras.models.load_model('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/CNN-DBN/DBN.pkl")
            return model
        
    elif (feature_type == 4) & (classification_type == 0):
           # model= {}
           # model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithIP Top11/Binary/CNN-DBN/DBN.pkl")
            return model
    elif (feature_type == 4) & (classification_type == 1):
            #model= {}
            #model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithIP Top11/Multi-class/CNN-DBN/DBN.pkl")
            return model
        
    elif (feature_type == 5) & (classification_type == 0):
            #model= {}
            #model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithoutIP Top9/Binary/CNN-DBN/DBN.pkl")
            return model

    elif (feature_type == 5) & (classification_type == 1):
           # model= {}
            #model['main'] = tf.keras.models.load_model('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-DBN/CNN')
            model = dbn.SupervisedDBNClassification.load("./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/CNN-DBN/DBN.pkl")
            return model

def getDBN(feature_type, classification_type):
    
    if (feature_type == 0) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load("./Resources/models/Binary/With IP/DBN/DBN.pkl")
        return model
    elif (feature_type == 0) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/Multi-class/With IP/DBN/DBN.pkl')
        return model
    elif( feature_type == 1) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/Multi-class/Without IP/DBN/DBN.pkl')
        return model
    elif (feature_type == 1) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/Binary/Without IP/DBN/DBN.pkl')
        return model
    elif (feature_type == 2) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/DBN/DBN.pkl')
        return model
    elif (feature_type == 2) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/DBN/DBN.pkl')
        return model
    elif (feature_type == 3) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/DBN/DBN.pkl')
        return model
    elif (feature_type == 3) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/DBN/DBN.pkl')
        return model
    elif (feature_type == 4) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/IG feature selection/WithIP Top11/Binary/DBN/DBN.pkl')
        return model
    elif (feature_type == 4) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/IG feature selection/WithIP Top11/Multi-class/DBN/DBN.pkl')
        return model
    elif (feature_type == 5) & (classification_type == 0):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/IG feature selection/WithoutIP Top9/Binary/DBN/DBN.pkl')
        return model
    elif (feature_type == 5) & (classification_type == 1):
        model= {}
        model['main'] = dbn.SupervisedDBNClassification.load('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/DBN/DBN.pkl')
        return model
