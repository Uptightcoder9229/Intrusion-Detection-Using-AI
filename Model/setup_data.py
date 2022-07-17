import pandas as pd
import numpy as np
from datetime import datetime
import gc
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Flatten
from keras.models import Model
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

pd.set_option('display.max_columns', None)

def convert_bytes(df):
    if 'M' in df:
        df = df.split('M')
        df = df[0].strip()
        df = float(df) * 1000000
    elif 'B' in df:
        df = df.split('B')
        df = df[0].strip()
        df =  float(df) * 1000000000
    else: 
        df =float(df)
    return df
#Load the CIDDS-001 dataset
def load_dataset():
    a = pd.read_csv('./Resources/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv', low_memory=False, encoding='cp1252')
    b = pd.read_csv('./Resources/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week2.csv', low_memory=False, encoding='cp1252')
    c =  pd.read_csv('./Resources/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week2.csv', low_memory=False, encoding='cp1252')
    d =  pd.read_csv('./Resources/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week3.csv', low_memory=False, encoding='cp1252')
    e =  pd.read_csv('./Resources/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv', low_memory=False, encoding='cp1252')

    #Drop Normal Traffic from dataset
    b.drop(b[b['attackType'] == '---'].index, axis = 0, inplace= True)  
    c.drop(c[c['attackType'] == '---'].index, axis = 0, inplace= True)  
    d.drop(d[d['attackType'] == '---'].index, axis = 0, inplace= True)   

    data_external = pd.concat([c,d,e], axis = 0)
    data_external.reset_index(drop= True, inplace= True)

    #to Increment attackID values
    data_external['attackID'] = data_external['attackID'].apply(lambda x: str(int(x) + 70) if x != '---' else x)

    #Join all week data
    data = pd.concat([a,b,data_external], axis = 0)
    data.reset_index(drop= True, inplace= True)
    # Convert String value Bytes to Numeric
    data['Bytes'] = data['Bytes'].apply(lambda x: convert_bytes(x))
    columns = ['Src Pt', 'Dst Pt','Tos','Flows','Packets', 'Bytes']
    #Convert rows to Numeric
    for i in columns:
        data[i] = pd.to_numeric(data[i])
    del columns
    del a,b,c,d,e, data_external
    gc.collect()
    return data

#Load the CIDDS-002 dataset
def load_dataset_2():
    a = pd.read_csv('./Resources/CIDDS-002/traffic/week1.csv', low_memory=False, encoding='cp1252')
    b = pd.read_csv('./Resources/CIDDS-002/traffic/week2.csv', low_memory=False, encoding='cp1252')

    data = pd.concat([a,b], axis = 0)
    data.reset_index(drop= True, inplace= True)
    data['Bytes'] = data['Bytes'].apply(lambda x: convert_bytes(x))
    columns = ['Src Pt', 'Dst Pt','Tos','Flows','Packets', 'Bytes']
    for i in columns:
        data[i] = pd.to_numeric(data[i]);
    del columns
    del a,b
    gc.collect()
    return data

#Converts Hexadecimal value to Binary
def hex_to_binary(hexdata):
    scale = 16 ## equals to hexadecimal
    num_of_bits = 9
    return bin(int(hexdata, scale))[2:].zfill(num_of_bits)

#Converts TCP flags to Binary
def to_Binary(x):
    l = 0
    x = '...' + x
    x = list(x)
    for i in x:
        if (i=='.'):
            x[l]= '0'
        else:
            x[l] = '1'
        l = l +1
    return ''.join(x)

#Converts the 'Flags' column to 9 indiviual columns (manual oneshot encoding)
def flag_convert(df):  
    hex_values = list(df[(df['Flags'].str.contains("0x", na=False))]['Flags'].unique())
    flag_values = list(df[~(df['Flags'].str.contains("0x", na=False))]['Flags'].unique())
    binary_values = {}
    for i in hex_values:
         binary_values[i] = (hex_to_binary(i))
    for i in flag_values:
         binary_values[i] = (to_Binary(i))
    temp = df['Flags'].replace(binary_values)
    temp = pd.DataFrame(temp.apply(list).tolist())
    temp.columns = ['N','C','E','U' ,'A','P','R','S','F']
    for i in temp.columns:
        temp[i] = pd.to_numeric(temp[i])
    temp = temp.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, temp], axis = 1)
    return df

#make a IP_pairs 
def make_pair(df):
    ip_pair = df['Src IP Addr'] +'/' +df['Dst IP Addr']
    source_ip = df['Src IP Addr'].unique().tolist()
    destination_ip = df['Dst IP Addr'].unique().tolist()
   # df = df.drop(columns = ['Src IP Addr', 'Dst IP Addr'])
    df.insert(1, ' IP Pair', ip_pair)
    return df

def check_inverse(df):
    list_pairs = df[' IP Pair'].unique()
    tuple_pair = []
    for i in list_pairs:
        tuple_pair.append(tuple((i.split('/'))))
    dic_store = {}
    for i in tuple_pair:
        if (i  not in dic_store.keys()) and (i[::-1] not in dic_store.keys()):
            dic_store[i] = i[0] + '/' +i[1]
    dic_final = {}
    for i in dic_store.keys():
        dic_final[i[0] + '/' +i[1]] = dic_store[i]
        dic_final[i[1] + '/' +i[0]] = dic_store[i]
    df[' IP Pair'] = df[' IP Pair'].map(dic_final)               
    return df
# Normalize all numeric rows using MinMaxScaler
def normalize(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    columns = df.select_dtypes(include=numerics).columns
    normalized = df[columns]
    transformed = MinMaxScaler().fit(normalized).transform(normalized)
    transformed = pd.DataFrame(transformed)
    j = 0
    col = {}
    for i in columns:
        col[j] = i
        j=j+1
    transformed = transformed.rename(columns = col)
    transformed = transformed.reset_index()
    for i in columns:
        df[i] = transformed[i].to_numpy()
    return df

def normalize_IP(df):
    columns = ['sourceIP_feature 1', 'sourceIP_feature 2', 'sourceIP_feature 3', 'sourceIP_feature 4', 'destIP_feature 1',
              'destIP_feature 2', 'destIP_feature 3', 'destIP_feature 4']
    normalized = df[columns]
    transformed = MinMaxScaler().fit(normalized).transform(normalized)
    transformed = pd.DataFrame(transformed)
    j = 0
    col = {}
    for i in columns:
        col[j] = i
        j=j+1
    transformed = transformed.rename(columns = col)
    transformed = transformed.reset_index()
    for i in columns:
        df[i] = transformed[i].to_numpy()
    return df
    
def one_shot_withoutIP(df):
    label_encoder = LabelEncoder()
    df['attackType'] = label_encoder.fit_transform(df['attackType'])
    print(list(label_encoder.classes_))
    print(list(label_encoder.transform(label_encoder.classes_)))

    df['Proto'] = label_encoder.fit_transform(df['Proto'])
    print(list(label_encoder.classes_))
    print(list(label_encoder.transform(label_encoder.classes_)))
    
    onehot_encoder1 = OneHotEncoder()
    onehot_encoder1.fit(df.Proto.to_numpy().reshape(-1, 1))
    proto = onehot_encoder1.transform(df.Proto.to_numpy().reshape(-1, 1))
    proto = pd.DataFrame.sparse.from_spmatrix(proto)
    proto.astype('int32')
    proto.columns = label_encoder.classes_
   # print(proto.head(1))
    df = pd.concat([df, proto], axis = 1)
    return df

def one_shot_withIP(df):
    label_encoder = LabelEncoder()
    df['attackType'] = label_encoder.fit_transform(df['attackType'])
    print(list(label_encoder.classes_))
    print(list(label_encoder.transform(label_encoder.classes_)))
    
    df['sourceIP_feature 1'] = label_encoder.fit_transform(df['sourceIP_feature 1'])
    df['sourceIP_feature 2'] = label_encoder.fit_transform(df['sourceIP_feature 2'])
    df['sourceIP_feature 3'] = label_encoder.fit_transform(df['sourceIP_feature 3'])
    df['sourceIP_feature 4'] = label_encoder.fit_transform(df['sourceIP_feature 4'])
    
    
    df['destIP_feature 1'] = label_encoder.fit_transform(df['destIP_feature 1'])
    df['destIP_feature 2'] = label_encoder.fit_transform(df['destIP_feature 2'])    
    df['destIP_feature 3'] = label_encoder.fit_transform(df['destIP_feature 3'])
    df['destIP_feature 4'] = label_encoder.fit_transform(df['destIP_feature 4'])
        
    df['Proto'] = label_encoder.fit_transform(df['Proto'])
    print(list(label_encoder.classes_))
    print(list(label_encoder.transform(label_encoder.classes_)))
    
    onehot_encoder1 = OneHotEncoder()
    onehot_encoder1.fit(df.Proto.to_numpy().reshape(-1, 1))
    proto = onehot_encoder1.transform(df.Proto.to_numpy().reshape(-1, 1))
    proto = pd.DataFrame.sparse.from_spmatrix(proto)
    proto.astype('int32')
    proto.columns = label_encoder.classes_
    df = pd.concat([df, proto], axis = 1)
    return df
#Drop columns
def drop_columns(df):
    if 'label' in df.columns:
        return df.drop(columns = ['Date first seen', 'Flows', 'label', 'attackID','Flags',
                              'attackDescription', 'Src IP Addr', 'Dst IP Addr','Proto'], axis =1)
    else:
        return df.drop(columns = ['Date first seen', 'Flows', 'class', 'attackID','Flags',
                              'attackDescription', 'Src IP Addr', 'Dst IP Addr','Proto'], axis =1)

#Split IP address into features, 7 features
def split_to_net(IP_address):
    IP_list = IP_address.split(".")
    needed_len = 7
    needed_len = needed_len - len(IP_list)
    for i in range(0,needed_len,1):
        IP_list.append('0')
    return IP_list
#replace unknown IP address, and convert to columns
def IP_split(df): 
    replace = {"ATTACKER1":"0.0.0.0",
           "ATTACKER2":"0.0.0.0",
           "ATTACKER3":"0.0.0.0",
           "EXT_SERVER": "0.0.0.0.1",
          "OPENSTACK_NET": "0.0.0.0.0.1",
          "DNS": "0.0.0.0.0.0.1"}
    df = df.replace({"Src IP Addr": replace, "Dst IP Addr": replace}, value=None)
    temp_source = df["Src IP Addr"].apply(lambda x: "0.0.0.0.0.0.0" if ('_') in x else x)
    temp_des = df['Dst IP Addr'].apply(lambda x: "0.0.0.0.0.0.0" if ('_') in x else x)
    #for Source IP
    temp_source = temp_source.apply(lambda x: split_to_net(x) )
    temp_source = pd.DataFrame(temp_source.apply(list).tolist())
    temp_source.columns = ['sourceIP_feature 1','sourceIP_feature 2','sourceIP_feature 3','sourceIP_feature 4' ,
                    'sourceEXT_SERVER','sourceOPENSTACK_NET','sourceDNS']
    for i in temp_source.columns:
        temp_source[i] = pd.to_numeric(temp_source[i])
    temp_source = temp_source.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, temp_source], axis = 1)
    #for Destination IP
    temp_des = temp_des.apply(lambda x: split_to_net(x) )
    temp_des = pd.DataFrame(temp_des.apply(list).tolist())
    temp_des.columns = ['destIP_feature 1','destIP_feature 2','destIP_feature 3','destIP_feature 4' ,
                    'destEXT_SERVER','destOPENSTACK_NET','destDNS']
    for i in temp_des.columns:
        temp_des[i] = pd.to_numeric(temp_des[i])
    temp_des = temp_des.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, temp_des], axis = 1)
    return df
#Convert Time stamp to group normal traffic
def unix_time(df):
    df['Date first seen'] = df['Date first seen'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f'))
    df['Date first seen'] = df['Date first seen'].apply(lambda x: x.timestamp()*1000)
    return df

def normal_profile(grouped):
    grouped_data = grouped
    grouped['---'] = unix_time(grouped['---'])
    start_time = int(grouped['---'].head(1)['Date first seen'].values[0])
    end_time = int(grouped['---'].tail(1)['Date first seen'].values[0])
    normal_data = dict(tuple( grouped['---'].groupby( pd.cut(
            grouped['---']['Date first seen'],
               np.arange(start_time, end_time, 3*3600000)))))
    del grouped['---']
    num = []
    for i in grouped_data.keys():
          num.append(len(grouped_data[i]))
    num = max(num)
    grouped = {**grouped, **normal_data}
    return grouped, num
#Deletes Large instances; this is done due to hardware restrictions
def del_largeInstances(dic, length):
    remove_ID = []
    for i in dic.keys():
        if (i != '---'):
            if(len(dic[i]) >= length):
                remove_ID.append(i)
    removed_attacks = {}
    for i in remove_ID:
        removed_attacks[i] = dic[i]
        del dic[i]
    return dic

def make_4D(arr):
    x = []
    for i in range(0, len(arr),1):
        temp = []
        for j in range(0,len(arr[i]),1):
             temp.append([np.array([k]) for k in arr[i][j]])
        x.append(np.array(temp).astype(np.float32))
    return np.array(x).astype(np.float32)

def roundup(x):
    return x if x % 100 == 0 else x + 100 - x % 100
#Convert to 3D arrays, input dict
def make_array(dic, num):
    x = []
    y = []
    zero_arrays = []
    for i in dic.keys():
        if ( len(dic[i]) == 0):
            zero_arrays.append(i)
    for i in zero_arrays:
        del dic[i]
    for i in dic.keys():
        x.append(np.array(dic[i].drop(['attackType'],axis = 1)).astype(np.float32))
        y.append(dic[i]['attackType'].values[0])
    features = len(x[1][1])
    o = roundup(num)
    index = 0
    for i in x:
        l = len(i)
        i = list(i)
        if(o > l):
            l = o-l
            for j in range(0, l, 1):
                i.append([0] * features)
        elif (o<l):
            l = l-o
            i = i[:-l]
        
        x[index] = np.array(i).astype(np.float32)
        index = index + 1
    return x,y

def dataprep_WithIP(data):
    data = IP_split(data)
    data = normalize(data)
    if 'label' in data.columns:
        data['GRE  '] = 0.0
    data =  one_shot_withIP(data)
    
    data = normalize_IP(data)
    return data

def dataprep_WithoutIP(data):
    data = normalize(data)
    if 'label' in data.columns:
        data['GRE  '] = 0.0
    data =  one_shot_withoutIP(data)
    
    return data   

def preprocess_data(data, feature_type, classification_type):
    if classification_type == 0:
        data['attackType'] = data['attackType'].apply(lambda x:  'attack' if (x!= '---') else x )
    
    if (feature_type == 0) | (feature_type == 2)| (feature_type == 4):
        data = dataprep_WithIP(data)
    elif (feature_type == 1) | (feature_type == 3) | (feature_type == 5):
        data = dataprep_WithoutIP(data)
    #print(data.head())
    if classification_type == 1:
        if 'label' in data.columns:
            data['attackType'] = data['attackType'].replace(1, 4 )
    grouped_data= dict(tuple(data.groupby(['attackID'])))
    del data
    gc.collect()
    grouped_data = del_largeInstances(grouped_data, 20000)
    for i in grouped_data.keys():
        grouped_data[i] = flag_convert(grouped_data[i])
    grouped_data, num = normal_profile(grouped_data)
    num = 19800
    for i in grouped_data.keys():
        grouped_data[i] =  drop_columns(grouped_data[i])
    
    if feature_type == 2:
        #Columns for CFS With IP - 7
        cfs_rows = ['F','S','sourceIP_feature 4','destIP_feature 4','Duration','sourceIP_feature 1','Packets', 'attackType']
        for i in grouped_data.keys():
            if ( len(grouped_data[i]) != 0):
                grouped_data[i] = grouped_data[i][cfs_rows]
    elif feature_type == 3:
        #Columns for CFS Without IP- 6
        selected_features = ['S', 'R', 'Packets', 'Tos', 'Duration', 'P', 'attackType']
        for i in grouped_data.keys():
            if ( len(grouped_data[i]) != 0):
                grouped_data[i] = grouped_data[i][selected_features]
    elif feature_type == 4:
        #Columns for IG with IP - 11
        selected_features = ['Bytes', 'sourceIP_feature 4', 'Src Pt', 'Dst Pt', 'destIP_feature 4',
            'Duration', 'Packets', 'S', 'F', 'sourceIP_feature 3',
            'destIP_feature 3', 'attackType']
        for i in grouped_data.keys():
            if ( len(grouped_data[i]) != 0):
                grouped_data[i] = grouped_data[i][selected_features]
    elif feature_type == 5:
        #Columns for IG without IP - 9
        selected_features = ['Bytes', 'Src Pt', 'Dst Pt', 'Duration', 'Packets', 'Tos', 'TCP  ',
            'UDP  ', 'ICMP ', 'attackType']
        for i in grouped_data.keys():
            if ( len(grouped_data[i]) != 0):
                grouped_data[i] = grouped_data[i][selected_features]

    X , Y = make_array(grouped_data, num)
    return np.array(X), np.array(Y)

def flatten(X):
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    return X


def test_split(X, Y, split = 20):
     X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size= (split/100), random_state=0,  stratify=Y)
     return X_test, Y_test

def test_load():
    data = load_dataset()
    data = make_pair(data)
    data = check_inverse(data)
    #data = unix_time(data)
    X, Y = preprocess_data(data, 0, 0)
    return True


def plot_confusion_matrix(data, labels, axes):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
 
    """

    data_df = pd.DataFrame(data,
                     index = labels, 
                     columns = labels)
    
    sns.heatmap(data_df, annot=True,  cbar= False,ax = axes, robust = True)
    axes.set_title('Confusion Matrix')
    axes.tick_params(axis='x', labelrotation = 45)
    axes.tick_params(axis='y', labelrotation = 45)
    return axes

def plot_classification_report(classificationReport, axes,
                               title='Classification report',
                               cmap='RdBu'):
    sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True,  cbar= False, robust = True,cmap= cmap, ax = axes)
    axes.set_title(title)
    axes.tick_params(axis='x', labelrotation = 45)
    axes.tick_params(axis='y', labelrotation = 45)
    return axes

def perf_measure(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = sum(FP.astype(float))
    FN = sum(FN.astype(float))
    TP = sum(TP.astype(float))
    TN = sum(TN.astype(float))
    return TP, FP, TN, FN
def test_single_model(model, X_test, Y_test, classification_type, axes):
    
    Y_pred = model.predict(X_test)
    performace_paramters = {}
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test, Y_pred)
    performace_paramters['Accuracy'] = accuracy
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, Y_pred, average='macro')
    performace_paramters ['Precision'] = precision
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, Y_pred,average='macro')
    performace_paramters['Recall'] = recall
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, Y_pred, average='macro')
    performace_paramters['F1 score'] = f1
    kappa = cohen_kappa_score(Y_test, Y_pred)
    performace_paramters['Cohens kappa'] = kappa
    
    
    fpr = {}
    tpr = {}
    thresh ={}
    n_class = 5
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(Y_test, Y_pred, pos_label=i)
    
    # plotting    
    
    #plt.savefig('Multiclass ROC',dpi=300); 
    if classification_type == 0:
        labels = [0,1]
        target_names = ['Normal', 'Attack']
        axes.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Normal vs Rest')
        axes.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Attack vs Rest')
        axes.set_title('Multiclass ROC curve')
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive rate')
        axes.legend(loc='best')
    else:
        labels = [0,1,3,4]
        target_names = ['Normal', 'BruteForce', 'PingScan', 'PortScan']
        axes.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Normal vs Rest')
        axes.plot(fpr[1], tpr[1], linestyle='--',color='green', label='BruteForce vs Rest')
        axes.plot(fpr[3], tpr[3], linestyle='--',color='red', label='PingScan vs Rest')
        axes.plot(fpr[4], tpr[4], linestyle='--',color='black', label='PortScan vs Rest')
        axes.set_title('Multiclass ROC curve')
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive rate')
        axes.legend(loc='best')
    matrix = confusion_matrix(Y_test, Y_pred, labels)

    TP, FP, TN, FN = perf_measure(matrix) 
    performace_paramters['True Positive'] = TP
    performace_paramters['True Negative'] = TN
    performace_paramters['False Postive'] = FP
    performace_paramters['False Negative'] = FN  

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    performace_paramters['Sensitivity/True Positive Rate'] = TPR
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    performace_paramters['Specificity/True Negative Rate'] = TNR
    # Negative predictive value
    NPV = TN/ float(TN+FN)
    performace_paramters['Negative Predictive Value'] = NPV
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    performace_paramters['Fall out/False Positive Rate'] = FPR
    # Miss 0r False negative rate
    FNR = FN/float(TP+FN)
    performace_paramters['Miss/False Negative rate'] = FNR
    # False discovery rate
    FDR = FP/float(TP+FP)
    performace_paramters['False discovery rate'] = FDR

    #plot_matrix = plot_confusion_matrix(matrix, labels)
    clf_report = classification_report(Y_test,
                                   Y_pred,
                                   labels=labels,
                                   target_names=target_names,
                                   output_dict=True)
    return performace_paramters, axes, matrix, clf_report

def plot_deep_model(model, axe):
    plot_model(model, show_shapes=True, to_file='./Resources/model.png')
    img = plt.imread("./Resources/model.png")
    axe.get_xaxis().set_visible(False)
    axe.get_yaxis().set_visible(False)
    axe =axe.imshow(img)
    #axe.title('Model')
    return axe

def test_deep_model(model, X_test, Y_test, classification_type, axes):
    #graph = tf.compat.v1.get_default_graph()
    #with graph.as_default():
    tf.compat.v1.enable_eager_execution()
    Y_pred = model.predict(X_test.astype('float32'), verbose=0)
    Y_pred = np.argmax(Y_pred,axis=1)
    performace_paramters = {}
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test, Y_pred)
    performace_paramters['Accuracy'] = accuracy
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, Y_pred, average='macro')
    performace_paramters ['Precision'] = precision
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, Y_pred,average='macro')
    performace_paramters['Recall'] = recall
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, Y_pred, average='macro')
    performace_paramters['F1 score'] = f1
    kappa = cohen_kappa_score(Y_test, Y_pred)
    performace_paramters['Cohens kappa'] = kappa

    fpr = {}
    tpr = {}
    thresh ={}
    n_class = 5
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(Y_test, Y_pred, pos_label=i)
    
    # plotting    
    
    #plt.savefig('Multiclass ROC',dpi=300); 
    if classification_type == 0:
        labels = [0,1]
        target_names = ['Normal', 'Attack']
        axes.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Normal vs Rest')
        axes.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Attack vs Rest')
        axes.set_title('Multiclass ROC curve')
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive rate')
        axes.legend(loc='best')
    else:
        labels = [0,1,3,4]
        target_names = ['Normal', 'BruteForce', 'PingScan', 'PortScan']
        axes.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Normal vs Rest')
        axes.plot(fpr[1], tpr[1], linestyle='--',color='green', label='BruteForce vs Rest')
        axes.plot(fpr[3], tpr[3], linestyle='--',color='red', label='PingScan vs Rest')
        axes.plot(fpr[4], tpr[4], linestyle='--',color='black', label='PortScan vs Rest')
        axes.set_title('Multiclass ROC curve')
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive rate')
        axes.legend(loc='best')
    matrix = confusion_matrix(Y_test, Y_pred, labels)

    TP, FP, TN, FN = perf_measure(matrix) 
    performace_paramters['True Positive'] = TP
    performace_paramters['True Negative'] = TN
    performace_paramters['False Postive'] = FP
    performace_paramters['False Negative'] = FN  

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    performace_paramters['Sensitivity/True Positive Rate'] = TPR
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    performace_paramters['Specificity/True Negative Rate'] = TNR
    # Negative predictive value
    NPV = TN/ float(TN+FN)
    performace_paramters['Negative Predictive Value'] = NPV
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    performace_paramters['Fall out/False Positive Rate'] = FPR
    # Miss 0r False negative rate
    FNR = FN/float(TP+FN)
    performace_paramters['Miss/False Negative rate'] = FNR
    # False discovery rate
    FDR = FP/float(TP+FP)
    performace_paramters['False discovery rate'] = FDR

    #plot_matrix = plot_confusion_matrix(matrix, labels)
    clf_report = classification_report(Y_test,
                                   Y_pred,
                                   labels=labels,
                                   target_names=target_names,
                                   output_dict=True)
    return performace_paramters, axes, matrix, clf_report 



def get_feature_extractor(model, layer_name):
    try:
        layer = model.get_layer(layer_name)
    except:
        try:
            layer = model.get_layer('flatten_1')
        except:
            layer = model.get_layer('flatten_2')
            
    bottom_input = Input(model.input_shape[1:])
    bottom_output = bottom_input
    top_input = Input(layer.output_shape[1:])
    top_output = top_input

    bottom = True
    for layer in model.layers:
        if bottom:
            bottom_output = layer(bottom_output)
        else:
            top_output = layer(top_output)
        if layer.name == layer_name:
            bottom = False

    bottom_model = Model(bottom_input, bottom_output)

    return bottom_model

def feature_extractor_set(model, X_test):
    X_ext = model.predict(X_test)
    return X_ext

def make_text(model):
        text = model.get_config()
        display_text = ''
        for i in text.keys():
            if i =='layers':
                display_text = "Layers in Model: \n" 
                for j in text[i]:
                    for k in j:
                        if k == 'class_name':
                            display_text = display_text +'  Layer: ' + j[k] +" \n" 
                        elif k == 'config':
                            display_text = display_text +'      Hyper Parameters: \n'
                            for l in j[k].keys():
                                if not(isinstance(j[k][l], dict)):
                                    display_text =  display_text + f"        {l}: {j[k][l]} \n"
        text = model.optimizer.get_config()
        display_text = display_text + "Optimizer Hyper-Parameters: \n"
        for i in text.keys():
            display_text = display_text + "  "+ str(i) + ": " + str(text[i]) + " \n"

        return display_text