'''This file is for blah blah blah'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write("Hello World!")
st.write("# Heart Disease Prediction")
st.write("## ***with Random Forest Model***")

if st.button("Click me"):
    st.write("You clicked me.")
else:
    st.write("You DID NOT click me!")

df = pd.read_csv("./data/heart_statlog_cleveland_hungary_final.csv")
st.dataframe(df[1:5])
st.write("Above you can see an exaple from the data")

question_sex = st.radio(
    "Sex",
    ["male", "female"],
    index=None,
)

question_age = st.number_input(
    "Age", value=None, placeholder="Type a number...", min_value= 18, max_value= 110
)

fig, ax = plt.subplots()
ax.hist(df['age'], bins=20)

st.pyplot(fig)
