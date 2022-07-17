
from sklearn import metrics
from Model.setup_data import *
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#0: Binary 1: Multiclass
#feature_type : 0: ALL 1: 20S 2:7 features CFS 3: 6 features CFS  4: 11 features 1g  5: 9 features IG

def getKNN(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return pickle.load(open("./Resources/models/Binary/With IP/KNN/KNN.sav", 'rb'))    
    elif (feature_type == 0) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/With IP/KNN/KNN.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/Without IP/KNN/KNN.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 0):
        return pickle.load(open('./Resources/models/Binary/Without IP/KNN/KNN.sav', 'rb'))
    
    elif (feature_type == 2) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/KNN/KNN.sav', 'rb'))
    elif (feature_type == 2) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/KNN/KNN.sav', 'rb'))

    elif (feature_type == 3) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/KNN/KNN.sav', 'rb'))
    elif (feature_type == 3) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/KNN/KNN.sav', 'rb'))
    
    elif (feature_type == 4) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Binary/KNN/KNN.sav', 'rb'))
    elif (feature_type == 4) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Multi-class/KNN/KNN.sav', 'rb'))
    
    elif (feature_type == 5) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Binary/KNN/KNN.sav', 'rb'))
    elif (feature_type == 5) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/KNN/KNN.sav', 'rb'))

def getDT(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return pickle.load(open("./Resources/models/Binary/With IP/DT/DT.sav", 'rb'))
    elif (feature_type == 0) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/With IP/DT/DT.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/Without IP/DT/DT.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 0):
        return pickle.load(open('./Resources/models/Binary/Without IP/DT/DT.sav', 'rb'))

    elif (feature_type == 2) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/DT/DT.sav', 'rb'))
    elif (feature_type == 2) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/DT/DT.sav', 'rb'))

    elif (feature_type == 3) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/DT/DT.sav', 'rb'))
    elif (feature_type == 3) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/DT/DT.sav', 'rb'))
    
    elif (feature_type == 4) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Binary/DT/DT.sav', 'rb'))
    elif (feature_type == 4) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Multi-class/DT/DT.sav', 'rb'))
    
    elif (feature_type == 5) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Binary/DT/DT.sav', 'rb'))
    elif (feature_type == 5) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/DT/DT.sav', 'rb'))

def getRF(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return pickle.load(open("./Resources/models/Binary/With IP/RF/RF.sav", 'rb'))
    elif (feature_type == 0) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/With IP/RF/RF.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/Without IP/RF/RF.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 0):
        return pickle.load(open('./Resources/models/Binary/Without IP/RF/RF.sav', 'rb'))

    elif (feature_type == 2) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/RF/RF.sav', 'rb'))
    elif (feature_type == 2) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/RF/RF.sav', 'rb'))

    elif (feature_type == 3) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/RF/RF.sav', 'rb'))
    elif (feature_type == 3) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/RF/RF.sav', 'rb'))
    
    elif (feature_type == 4) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Binary/RF/RF.sav', 'rb'))
    elif (feature_type == 4) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Multi-class/RF/RF.sav', 'rb'))
    
    elif (feature_type == 5) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Binary/RF/RF.sav', 'rb'))
    elif (feature_type == 5) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/RF/RF.sav', 'rb'))

def getSVM(feature_type, classification_type):
    if (feature_type == 0) & (classification_type == 0):
        return pickle.load(open("./Resources/models/Binary/With IP/SVM/SVM.sav", 'rb'))
    elif (feature_type == 0) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/With IP/SVM/SVM.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 1):
        return pickle.load(open('./Resources/models/Multi-class/Without IP/SVM/SVM.sav', 'rb'))
    elif (feature_type == 1) & (classification_type == 0):
        return pickle.load(open('./Resources/models/Binary/Without IP/SVM/SVM.sav', 'rb'))

    elif (feature_type == 2) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Binary/SVM/SVM.sav', 'rb'))
    elif (feature_type == 2) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/With IP_ Top 7/Multi-class/SVM/SVM.sav', 'rb'))

    elif (feature_type == 3) & (classification_type == 0):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Binary/SVM/SVM.sav', 'rb'))
    elif (feature_type == 3) & (classification_type == 1):
        return pickle.load(open('./Resources/models/CFS feature selection/Without IP_Top 6/Multi-class/SVM/SVM.sav', 'rb'))
    
    elif (feature_type == 4) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Binary/SVM/SVM.sav', 'rb'))
    elif (feature_type == 4) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithIP Top11/Multi-class/SVM/SVM.sav', 'rb'))
    
    elif (feature_type == 5) & (classification_type == 0):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Binary/SVM/SVM.sav', 'rb'))
    elif (feature_type == 5) & (classification_type == 1):
        return pickle.load(open('./Resources/models/IG feature selection/WithoutIP Top9/Multi-class/SVM/SVM.sav', 'rb'))










