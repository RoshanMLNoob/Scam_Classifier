import numpy as np
import pandas as pd
import re

#=====Data_Appender=====

def add_data(data, files=r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\Data.csv",done=False):
    data = data.lower()
    dataset = re.findall(r'[a-zA-Z]+', data)
    df = pd.read_csv(files)
    num = np.max(df["Tokens"])
    with open(files,  "a") as f:
        for i in dataset:
            if i not in list(pd.read_csv(files)["Words"]):
                num += 1
                f.write(f"{num},{i}\n")
    return dataset
def vectorize(data, files=r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\Data.csv"):
    vector = []
    data = data.lower()
    dataset = re.findall(r'[a-zA-Z]+', data)
    df = pd.read_csv(files)
    scam_words = list(df["Words"])
    for i in (scam_words):
        vector.append(dataset.count(i))
    return vector

#=====Perceptron=====

def perceptron(D,T,th=None,th0=0):
    if th == None:
        th = np.array([0]*len(D[0][0]))
    th,th0 = th,th0
    for k in range(T):
        for i in range(len(D)):
            if (D[i][1])*(np.dot(np.array(D[i][0]),th)+th0) <= 0:
                th = th + (D[i][1] * np.array(D[i][0]))
                th0 = th0 + D[i][1]
    return [th.tolist(),th0]

#=====Saving_Hypothesis=====

def save_thetas(th,th0,file=r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\thetas.csv"):
    with open(file, "a") as f:
        f.write(f'"{th}",{th0}\n')
    return [th,th0,file]
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

#====Data Format===



"""
D would bt given in form of
D = [vectorize(email),{1,0,-1},
    vectorize(emial2),{1,0,-1},
    vectorize(email3),{1,0,0-1}]
"""

#=====Final_Check=====

def is_spam(data, vectorized=False):
    df = pd.read_csv(r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\thetas.csv")
    TH,TH0 = list(df["th"]),list(df["th0"])
    th,th0 = np.array((str(TH[-1])[1:len(TH[-1])-1]).split(",")).astype(float),TH0[-1]
    if vectorized == False:
        vector = vectorize(data)
    else:
        vector = data
    RESULT =  np.dot(np.array(vector),np.array(th)) + int(th0) #(th*vector)+th0
    if sign(RESULT) == -1:
        print(f"Given Data was not classified spam with {RESULT}")
    elif sign(RESULT) == 1:
        print(f"Given Data was classified as Spam with {RESULT}")
    else:
        print(f"Our Model could not give a clear enough Verdict if it was a Spam or Not \n It is a pretty close call...")
    return sign(RESULT)


#== Stuff On Data and Shi ====

def calculate_confusion_matrix(performance, actual):
    tp,tn,fp,fn = 0,0,0,0
    for perf, act in zip(performance, actual):
        if perf == True:
            if act == 1:
                tp += 1 
            elif act == -1:
                tn += 1  
        elif perf == False:
            if act == 1:
                fn += 1  
            elif act == -1:
                fp += 1
    return {"TP":tp,"TN":tn,"FP":fp,"FN":fn}

def clean_dataset():
    New_Dataset = []
    Confirmations = []
    df = pd.read_csv(r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\Testing_Dataset.csv")
    Label = np.array(df["Label"]).astype(int)
    Vectors = np.array(df["Vectorized_Mail"])
    for i in range(len(Label)):
        New_Dataset.append([(np.array(re.findall(r'[0-9]+',(Vectors[i]))).astype(int)) , int(Label[i])])
    for point in New_Dataset:
        Confirmations.append(is_spam(point[0], True)==point[1])
    return [Confirmations , Label]

def stuff_with_Confusion_Matrix(TP,TN,FP,FN):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    F1 = (2*precision*recall)/(precision+recall)
    return {"accuracy":round(accuracy*100, 3), "precision":round(precision*100, 3), "recall":round(recall*100 ,3), "F1 Score":round(F1*100, 3)}

#===Model performance on diff vals of T as OnT

On10000 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On2500 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On1000 = {'accuracy': 96.33, 'precision': 93.103, 'recall': 72.973, 'F1 Score': 81.818}
On100 = {'accuracy': 95.639, 'precision': 73.469, 'recall': 97.297, 'F1 Score': 83.721}
On40 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On40 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On30 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On20 = {'accuracy': 95.327, 'precision': 72.0, 'recall': 97.297, 'F1 Score': 82.759}
On10 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On5 = {'accuracy': 95.95, 'precision': 75.0, 'recall': 97.297, 'F1 Score': 84.706}
On1 = {'accuracy': 94.081, 'precision': 68.0, 'recall': 91.892, 'F1 Score': 78.161}
    
#====Debug Menu====

def check_diff_T(T=40):
    New_Dataset = []
    df = pd.read_csv(r"C:\Users\rosha\OneDrive\Desktop\Programs_\EMail_Spam_Classifier\Training_Dataset.csv")
    Label = np.array(df["Label"]).astype(int)
    Vectors = np.array(df["Vectorized_Mail"])
    for i in range(len(Label)):
        New_Dataset.append([(np.array(re.findall(r'[0-9]+',(Vectors[i]))).astype(int)) , int(Label[i])])
    THS = perceptron(New_Dataset,T)
    save_thetas(THS[0],THS[1])
    CD = clean_dataset()
    Prediction, Actually= CD[0], (CD[1])[5:-1]
    Confusion_Matrix = calculate_confusion_matrix(Prediction, Actually)
    print(stuff_with_Confusion_Matrix(Confusion_Matrix["TP"],Confusion_Matrix["TN"],Confusion_Matrix["FP"],Confusion_Matrix["FN"]))
#check_diff_T()

#====User_Interface====

if __name__ == "__main__":
    DaTa = input(f"Enter the data to be checked for Spam or Not Spam: \n")
    is_spam(DaTa)