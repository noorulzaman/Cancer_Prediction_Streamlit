Libraries Imported:

pandas
streamlit
numpy
pickle
matplotlib
sklearn.ensemble

Data Preprocessing:

-read Cancer_Data.csv
-drop two unwanted columns i.e. id & Unnamed 32
-separate features and labels
-Used LabelEncoder to encode output labels
-split the data into train and test data

Model Used:

-RandomForest model is used

Generate Pickle

-Save trained model to use it in our project.

App:
-Sliders in the sidebar to tune the features
-Table to present the tuned features
-Pie Chart to show the predicted output probabilities of cancer whether it is malignant or benign

