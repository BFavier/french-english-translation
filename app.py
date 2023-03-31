import streamlit as st
import pygmalion as ml

@st.cache_data
def load_model():
    return ml.neural_networks.TextTranslator.load("model.pth")

data_load_state = st.text('Loading model...')
model = load_model()
data_load_state.text('Loading model... Done!')

st.session_state.input = ""
st.session_state.translation = ""

st.title("Neural Machine Translation français → anglais")
input_text_area = st.text_area(label="Le text à traduire:", value=st.session_state.input, placeholder="J'ai toujours voulu être un oiseau ...", max_chars=10000, height=100)
translation_text_area = st.text_area(label="Traduction:", value=st.session_state, disabled=True)