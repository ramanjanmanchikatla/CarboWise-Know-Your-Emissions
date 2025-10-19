# app.py
import streamlit as st
import joblib
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Load model and column names
model = joblib.load('co2_emission_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🚗 CO₂ Emission Predictor")
st.markdown("Enter vehicle specifications to predict CO₂ emissions (g/km)")

# Input fields
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=1, step=1)
fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, step=0.1)
fuel_type = st.selectbox("Fuel Type", ['Z', 'D', 'E', 'N', 'X'])

# Encode fuel type
fuel_encoding = {
    'Fuel Type_D': 1 if fuel_type == 'D' else 0,
    'Fuel Type_E': 1 if fuel_type == 'E' else 0,
    'Fuel Type_N': 1 if fuel_type == 'N' else 0,
    'Fuel Type_X': 1 if fuel_type == 'X' else 0
}

# Format input for model
input_data = [engine_size, cylinders, fuel_consumption]
for col in model_columns[3:]:
    input_data.append(fuel_encoding.get(col, 0))

# Init Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")  # or llama3-8b-8192 for speed

# --- LLM EXPLANATION TEMPLATE ---
def explain_with_llm(engine_size, cylinders, fuel_consumption, fuel_encoding, prediction):
    prompt_template = PromptTemplate(
        input_variables=["engine_size", "cylinders", "fuel_consumption", "fuel_encoding", "prediction"],
        template="""
You are an automotive emissions expert. A user has submitted their car's specifications:

- Engine Size: {engine_size} L
- Cylinders: {cylinders}
- Fuel Consumption: {fuel_consumption} L/100km
- Fuel Type Encoding: {fuel_encoding}
- Predicted CO₂ Emission: {prediction} g/km

Please provide:
1. A short, clear assessment — is this emission good, average, or high?
2. A bullet-point explanation of which factors contributed to the prediction
3. A list of 3–4 practical suggestions to reduce emissions

Be direct, friendly, and use simple language.
"""
    )

    prompt = prompt_template.format(
        engine_size=engine_size,
        cylinders=cylinders,
        fuel_consumption=fuel_consumption,
        fuel_encoding=str(fuel_encoding),
        prediction=f"{prediction:.2f}"
    )

    response = llm.invoke(prompt)
    return response.content

# --- LLM FOLLOW-UP CHAT TEMPLATE ---
def ask_llm_question(engine_size, cylinders, fuel_consumption, fuel_encoding, prediction, user_question):
    prompt = f"""
You are a helpful car emissions assistant.

The user has a car with these specs:
- Engine Size: {engine_size} L
- Cylinders: {cylinders}
- Fuel Consumption: {fuel_consumption} L/100km
- Fuel Type Vector: {fuel_encoding}
- Predicted CO₂ Emission: {prediction:.2f} g/km

Now answer this follow-up question clearly and helpfully:
"{user_question}"
"""
    response = llm.invoke(prompt)
    return response.content
prediction1 = model.predict([input_data])[0]
# --- PREDICTION ---
if st.button("Predict CO₂ Emission"):
    prediction = model.predict([input_data])[0]
    st.session_state.prediction = prediction  # store in session
    st.success(f"✅ Predicted CO₂ Emission: {prediction:.2f} g/km")

    with st.spinner("🔍 Analyzing with AI..."):
        explanation = explain_with_llm(engine_size, cylinders, fuel_consumption, fuel_encoding, prediction)
        st.markdown("### 🤖 LLM Summary & Advice")
        st.info(explanation)

# --- USER Q&A SECTION ---
st.markdown("---")
st.subheader("💬 Ask Follow-Up Question")
user_question = st.text_input("Ask something like: 'How to reduce?', 'What fuel is better?', etc.")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("💡 Thinking..."):
            followup = ask_llm_question(engine_size, cylinders, fuel_consumption, fuel_encoding, prediction1, user_question)
            st.markdown("### 💡 AI Answer")
            st.success(followup)
