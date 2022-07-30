import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model = open('svm_file.pkl','rb')
sv = joblib.load(model)

st.title("Iris Classification")

setosa = Image.open('Images/setosa.jpg')
versicolor = Image.open('Images/versicolor.jpg')
virginica = Image.open('Images/virginica.jpg')

st.sidebar.title('Features')
parameter_list = ['Sepal Length (cm)','Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
par_input_values = []
default_values = ['0.0','0.0','0.0','0.0']
values = []

for parameter, parameter_df in zip(parameter_list,default_values):
    values = st.sidebar.slider(label = parameter, key = parameter, value = float(parameter_df),
    min_value = 0.0,max_value = 8.0, step = 0.1)
    par_input_values.append(values)


input_variables=pd.DataFrame([par_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

if st.button("Click here to Classify"):
    prediction = sv.predict(input_variables)
    if prediction == 0 :
        st.image(setosa)
    elif prediction == 1:
        st.image(versicolor)
    else:
        st.image(virginica)
