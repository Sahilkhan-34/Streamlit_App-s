import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('KNN with Hyperparameter Tuning & Evaluation')

# Upload a File
file = st.sidebar.file_uploader("**Upload Only Preprocessed CSV file**", type=['csv'])


if file is not None:
    df = pd.read_csv(file)

    # select target column
    target_column = st.selectbox("**Choose The Target Column**", options=df.columns)

    # input output column
    x = df.drop(columns=target_column)
    y = df[target_column]
    
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Hypereparameter 
    st.sidebar.header('**Tune the Hyperparameter of KNN**')
    n_neighbours = st.sidebar.slider("**N-Neighbours**", min_value=1, max_value=10, step=1, value=3)
    leaf_size = st.sidebar.slider("**Leaf-Size**", min_value=10, max_value=100, step=10, value=20)
    p = st.sidebar.slider("**P**", min_value=1, max_value=5, step=1, value=2)
    metrix = st.sidebar.selectbox("**Distance Matrix**", options=['minkowski','manhattan', 'euclidean', 'chebyshev', 'cosine'], index=0)

    # Load the model
    KNN = KNeighborsClassifier(n_neighbors=n_neighbours, leaf_size=leaf_size, p=p, metric=metrix, n_jobs=-1)

    # Model fit
    KNN.fit(x_train, y_train)

    # Calculate the prediction
    y_pred = KNN.predict(x_test)

    # Calculate the accuracy, precision, recall, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("Accuracy Score: ",accuracy)
    st.write("Precision Score: ",precision)
    st.write("Recall Score: ",recall)
    st.write("F1 Score: ",f1)

    # plot confusion metrix
    cm = confusion_matrix(y_test,y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=[False,True])
    display.plot()
    plt.grid(False)
    plt.show()
    st.pyplot()

    # plot Auc-Roc Curve
    y_pred_prob = KNN.predict_proba(x_test)[:,1]
    fpr, tpr, thershold = roc_curve(y_test, y_pred_prob)
    plt.plot([0,1],[0,1], color="red", lw=2, label="Average Model")
    plt.plot(fpr,tpr, color="yellow", lw=2, label="KNN Model with Hyperparameter")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic :ROC-AUC")
    plt.legend()
    plt.show()
    st.pyplot()

    #Area under the curve
    st.write("Computed Area Under the Curve (AUC)",(auc(fpr, tpr)))