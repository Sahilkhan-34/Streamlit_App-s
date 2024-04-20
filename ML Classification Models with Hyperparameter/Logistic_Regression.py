import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score,roc_auc_score,roc_curve,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score






st.title("Logistic Regression Hyperparameter Tuning & Evaluation")

# Upload a file
file = st.sidebar.file_uploader("**Upload Preprocessed CSV file Only**", type=['csv'])

if file is not None:
    df = pd.read_csv(file)

    #   select the target column
    # st.subheader("Select the Target Colunmn")
    target_col = st.selectbox('**Select the Target Colunmn First**',options=df.columns)

    # Input Output Columns
    x = df.drop(columns=target_col)
    y = df[target_col]

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Hyperparameter
    st.sidebar.subheader("Tune the Hyperparameter of Logistic Regression")

    C = st.sidebar.slider("**Regularization Strength (C)**", min_value=0.1, max_value=10.0, step=1.0, value=2.0)
    max_iter = st.sidebar.slider("**Maximum Number of Iterations**", min_value=100, max_value=500, step=100, value=200 )
    penalty = st.sidebar.selectbox("**Penalty**", options=["l1", "l2"], index=0)
    solver = st.sidebar.selectbox("**Solver**", options=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], index=0)
    
    


    # Create a object of Model
    LR = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, n_jobs=-1)

    # Fit the Model
    LR.fit(x_train,y_train)

    # Calculate prediction
    y_predict = LR.predict(x_test)

    # Matrix Calculation and Display
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test,y_predict)

    st.write("Accuracy Score is : ", accuracy)
    st.write("Precision Score is : ", precision)
    st.write("Recall Score is : ", recall)
    st.write("F1 Score is : ", f1)

    # Auc Roc Curve & Confusion Matrix
    cm = confusion_matrix(y_test, y_predict)
    display = ConfusionMatrixDisplay(cm,display_labels=[False,True])

    # fig, axs = plt.subplots(1,2,figsize=(6,8))
    
    display.plot()
    plt.grid(False)
    plt.show()
    st.pyplot()


    tn,fp,fn,tp = confusion_matrix(y_test,y_predict).ravel()
    st.write("True Positive",tp)
    st.write("True Negative", tn)
    st.write("False Positive", fp)
    st.write("False Negative",fn)
    


    # Roc Curve
    y_predict_prob = LR.predict_proba(x_test)[:,1]

    fpr,tpr,threshold = roc_curve(y_test, y_predict_prob)
    plt.plot([0,1],[0,1],color="red",lw=2,label="Average-model")
    plt.plot(fpr,tpr,color="yellow",lw=2,label="Logistic Regression Model with Hyperparameter")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic :ROC-AUC")
    plt.legend()
    plt.show()
    st.pyplot()

    # st.write("True Psitive Rate : ", tpr)
    # st.write("False Positive Rate : ",fpr)

    # Area Under the curve
    st.write("Computed Area Under the Curve : ", auc(fpr,tpr))
