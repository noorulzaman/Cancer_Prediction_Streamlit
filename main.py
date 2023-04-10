import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Cancer Prediction App
This App predicts the type of Cancer!
""")

st.sidebar.header("Cancer Input Features")

df = pd.read_csv('Cancer_Data.csv')
df = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
feature_values = {}
for feature in df.columns:
    feature_values[feature] = st.sidebar.slider(feature.capitalize().replace("_", " "),
                                                min_value=0.0,
                                                max_value=max(df[feature]),
                                                )

data = pd.DataFrame(feature_values,index=[0])
st.write(data)

model = pickle.load(open("Classifier.pkl","rb"))
st.subheader("Cancer Prediction")
labels = ["Malignant"," Benign"]
prob = model.predict_proba(data)
fig1, ax1 = plt.subplots()
ax1.pie(prob[0],autopct="%1.1f%%",labels=labels,colors=["skyblue","darkred"],shadow=True,startangle=90)
ax1.axis('equal')
fig1.set_facecolor("grey")
st.pyplot(fig1)
st.write()
