import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,accuracy_score, recall_score, f1_score,classification_report,roc_auc_score,roc_curve,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score






st.title("Random Forest With Hyperparameter Tuning & Evaluation")

# Upload a file
file = st.file_uploader("Upload CSV File", type=['csv'])

if file is not None:
    df = pd.read_csv(file)

    #   select the target column
    st.subheader("Select the Target Colunmn")
    target_col = st.selectbox("Choose Target Column", options=df.columns)

    # Input Output Columns
    x = df.drop(columns=target_col)
    y = df[target_col]

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Hyperparameter
    st.subheader("Hyperparameter of Random Forest")
    n_estimator = st.slider("Number Of Estimators", min_value=10, max_value=200, step=20, value=100)
    max_depth = st.slider("Maximum Depth of Tree", min_value=1, max_value=20, value=5)
    criterion = st.selectbox("Choose Criterion", options=['gini', 'entropy'], index=0)
    max_features = st.selectbox("Choose Max_Features", options=['auto','sqrt', 'log2'], index=0)
    random_state = st.slider("Choose Random_state", min_value=0, max_value=200, value=42)


    # Create a object of Model
    RF = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, criterion=criterion, max_features=max_features, random_state=random_state, n_jobs=-1)

    # Fit the Model
    RF.fit(x_train,y_train)

    # Calculate prediction
    y_predict = RF.predict(x_test)

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

    display.plot()
    plt.grid(False)
    plt.show()

    tn,fp,fn,tp = confusion_matrix(y_test,y_predict).ravel()
    st.write("True Negative", tn)
    st.write("False Positive", fp)
    st.write("False Negative",fn)
    st.write("True Positive",tp)


    # Roc Curve
    y_predict_prob = RF.predict_proba(x_test)[:,1]

    fpr,tpr,threshold = roc_curve(y_test, y_predict_prob)
    plt.plot(fpr,tpr)

    # st.write("True Psitive Rate : ", tpr)
    # st.write("False Positive Rate : ",fpr)

    # Area Under the curve
    st.write("Computed Area Under the Curve : ", auc(fpr,tpr))
