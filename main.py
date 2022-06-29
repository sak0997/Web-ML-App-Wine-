import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Приложение Для Предсказания Вина
""")

st.sidebar.header('Входные параметры пользователя')

def user_input_features():
    alcogol = st.sidebar.slider('Alcohol', 11.0, 14.8, 12.1)
    malic_acid = st.sidebar.slider('Malic Acid', 0.74, 5.80, 3.4)
    ash = st.sidebar.slider('Ash', 1.36, 3.23, 1.39)
    alcalinity_of_ash = st.sidebar.slider('Alcalinity of Ash', 10.6, 30.0, 20.1)
    magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0, 93.0)
    total_phenols = st.sidebar.slider('Total Phenols', 0.98, 3.88, 3.4)
    flavanoids = st.sidebar.slider('Flavanoids', 0.34, 5.08, 1.3)
    nonflavanoid_henols = st.sidebar.slider('Nonflavanoid Phenols', 0.13, 0.66, 0.2)
    proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58, 2.4)
    colour_intensity = st.sidebar.slider('Colour Intensity', 1.3, 13.0, 3.4)
    hue = st.sidebar.slider('Hue', 0.48, 1.71, 1.3)
    diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.00, 2.2)
    proline = st.sidebar.slider('Proline', 278, 1680, 500)
    data = {'Alcohol': alcogol,
            'Malic Acid': malic_acid,
            'Ash': ash,
            'Alcalinity of Ash': alcalinity_of_ash,
            'Magnesium': magnesium,
            'Total Phenols': total_phenols,
            'Flavanoids': flavanoids,
            'Nonflavanoid Phenols': nonflavanoid_henols,
            'Proanthocyanins': proanthocyanins,
            'Colour Intensity': colour_intensity,
            'Hue': hue,
            'OD280/OD315 of diluted wines': diluted_wines,
            'Proline': proline}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Входные параметры пользователя:')
st.write(df)

wine = datasets.load_wine()
X = wine.data
Y = wine.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Метки классов и соответствующий им индексный номер:')
st.write(wine.target_names)

st.subheader('Предсказание:')
st.write(wine.target_names[prediction])
st.write(prediction)

st.subheader('Вероятность:')
st.write(prediction_proba)