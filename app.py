
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Titanic Survival Prediction App")

st.sidebar.header("Enter Passenger Details")

def user_input():
    Pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.sidebar.selectbox("Sex", ["male", "female"])
    Age = st.sidebar.slider("Age", 1, 80, 30)
    SibSp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
    Fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
    Embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

    data = {
        "Pclass": Pclass,
        "Sex": 1 if Sex == "male" else 0,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": {"C": 0, "Q": 1, "S": 2}[Embarked]
    }

    return pd.DataFrame([data])

input_df = user_input()
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0][1]

st.subheader("Prediction Result")
st.write("**Survived**" if prediction == 1 else "**Did Not Survive**")

st.subheader("Survival Probability")
st.write(f"{prob:.2f}")
