#this program detects diabetes un=sing machine learning
#import the libraries

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import RandomFrorestClassifier

from PIL import  Image
import streamlit as st


#create a title

st.write("""
#Diabetes detection
If someone has diabetes
""")

#open and display an image

image = Image.open('gr_kolala.png')

st.image(image, caption='ML', use_column_width=True)

#get the data

df = pd.read_csv('diabetes.csv')
st.subheader('Data information')

st.dataframe(df)

#show statistics on the data

st.write(df.describe())
#show data as a chart

chart  = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y'

X = df.iloc[:, 0:8].values

Y = df.iloc[:, -1].values

#split the data set in to 75% training and 25% for testing
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)

#get the feature input from the user

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)

    bloodPressure = st.sidebar.slider('blood_preasure',0,122,72)
    skinThickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age = st.sidebar.slider('age',21,81,29)


    user_data = {
        'pregnancies':pregnancies,
        'glucose':glucose,
        'bloodPreasure':bloodPressure,
        'skinThickness':skinThickness,
        'insulin':insulin,
        'BMI':BMI,
        'DPF':DPF,
        'age':age
    }

    #transform data into a dataframe

    features = pd.DataFrame(user_data,index = [0])
    return  features



#store the users input into a variable

user_input = get_user_input()


#set a sub header and display users input

st.subheader('User Input:')
st.write(user_input)

#create and train the model

RandomFrorestClassifier = RandomForestClassifier()

RandomFrorestClassifier.fit(X_train,Y_train)


#show the models metrices

st.subheader('Model Test Accuracy Score:')


st.write(str(accuracy_score(Y_test,RandomFrorestClassifier.predict(X_test)) *100)+'%')


#store the models predictions in a variable
prediction = RandomFrorestClassifier.predict(user_input)


#set a sub header and display the classification

st.subheader('Classifiction:')

st.write(prediction)






