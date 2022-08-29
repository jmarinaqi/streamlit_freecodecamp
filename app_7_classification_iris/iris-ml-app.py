from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.write("""
# APL App
Esta app permitirá identificar agentes que tengan mayor probabilidad de retirarse voluntariamente de Atlantic QI
De esta manera se realizarán actividades que ayuden a los agentes a reencontrar su amor por AQI. :heart:
Mientras tanto, hagamos un ejemplo de lo que se puede hacer con unas pocas lineas de codigo:
""")

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df1=pd.read_excel(filename)
# else:
#     st.warning("you need to upload a csv or excel file.")
uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.dataframe(df)
    st.table(df)


st.sidebar.header('User Input Parameters')

def user_input_features():
    AHT = st.sidebar.slider('AHT', 4.3, 7.9, 5.4)
    AUS = st.sidebar.slider('AUS', 2.0, 4.4, 3.4)
    ADH = st.sidebar.slider('ADH', 1.0, 6.9, 1.3)
    Active_months = st.sidebar.slider('Active_months', 0.1, 2.5, 0.2)
    data = {'sepal_length': AHT,
            'sepal_width': AUS,
            'petal_length': ADH,
            'petal_width': Active_months}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
