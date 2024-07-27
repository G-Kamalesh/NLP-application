import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout='wide')

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

def back():
    st.session_state['page'] = 'Home'

def change():
    st.session_state['page'] = 'Model'

@st.cache_resource
def loading_model():
    with open(r"K:\data\ML Model\Toxic Tweet\text_model.pkl",'rb') as f:
        model = pickle.load(f)
    with open(r"K:\data\ML Model\Toxic Tweet\countvectorizer.pkl",'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

c1,c2,c3 = st.columns([0.3,0.6,0.1])
c2.title(":red[Toxic Tweet Prediction]")

model, vectorizer=loading_model()

if st.session_state['page'] == 'Home':
    co1,co2 = st.columns([0.1,0.9])
    co2.text("""The objective of this project is to develop a machine learning model and deploy it as a user-friendly web
application that predicts wheather the given tweet is toxic or not.
""")
    with co2:
        w1 = st.container(border=True)
        w1.text("""
    To achieve these objectives, the solution involves:""")

        w1.write(":violet[Text Cleaning]") 
        w1.text("""Tokenized the sentence to words then convert it to lower case after that remove punctuations, digits,
special characters and stop words.""")
        w1.write(":violet[Text pre-processing]") 
        w1.text("Lemmatized the words to root form then store it and")
        w1.write(":violet[Feature Engineering]")
        w1.text("Feature encode the lemmatized words using either Count vectorizer or Tf-Idf Vectorizer")
        w1.write(":violet[Model Building]") 
        w1.text("Training and evaluating binary models for prediction.")
        w1.write(":violet[Model Deployment]")
        w1.text("""Developing a Streamlit application for real-time predictions, enabling users to input relevant data
and receive accurate prediction.""")

    with co2:
        m1,m2,m3,m4,m5 = st.columns([0.2,0.2,0.2,0.2,0.5],gap='small')
        m2.link_button("Linkedin Profile","https://www.linkedin.com/in/g-kamaleashwar-28a2802ba/")
        #m3.link_button("Hugging Face","https://huggingface.co/spaces/kamalesh-g/Singapore-RealEstate-Streamlit")
        m4.button("Launch ML Model",on_click=change)


if st.session_state['page'] == 'Model':
    c1.button("Back",on_click=back)
    b = st.text_input("Paste your tweet")
    v = st.button("Predict")
    if v:
        n = vectorizer.transform([b])
        c = model.predict(n)
        if c == 1:
            st.error("The given tweet is toxic")
        else:
            st.success("The given tweet is not toxic")