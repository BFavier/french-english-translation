import streamlit as st
import pygmalion as ml
import re
import string
from typing import List


@st.cache_resource(show_spinner=False)
def download_model():
    # return ml.neural_networks.TextTranslator.load("model.pth")
    return ml.neural_networks.TextTranslator.load("https://drive.google.com/file/d/1l4jxLJpLt8xmxf9JMIVM4lYv42erF--0/view?usp=share_link")


def format_sentences(text: str) -> List[str]:
    """
    Split an input text into formated input sentences
    """
    sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    sentences = (s.capitalize().strip() for s in sentences)
    sentences = (s.replace(" ,", ",").replace(" .", ".") for s in sentences)
    sentences = (re.sub("(?<=[^\\s\\^]):", " :", re.sub("(?<=[^\\s\\^])\?", " ?", re.sub("(?<=[^\\s\\^])!", " !", s))) for s in sentences)
    sentences = [s if s.endswith((".", "?", "!", ":")) else s+"." for s in sentences]
    return sentences


with st.spinner(text="Téléchargement du modèle ..."):
    model = download_model()

st.title("Traduction français → anglais")
input_text = st.text_area(label="Le text à traduire:", placeholder="J'ai toujours voulu être un vrai petit garçon ...", max_chars=10000, height=100)

if len(input_text) > 0:
    inputs = format_sentences(input_text)
    predictions = []
    progress_bar = st.progress(0., "Traduction en cours ...")
    for i, input in enumerate(inputs, start=1):
        predictions.append(model.predict(input, n_beams=3)[0])
        progress_bar.progress(int(i*100/len(inputs)), "Traduction en cours ...")
    predictions = ["".join(filter(lambda x: x in string.printable, s)) for s in predictions]  # Filtre les charactères non-affichables
    progress_bar.empty()
    st.caption("Traduction:")
    translation = " ".join(predictions)
    st.markdown(f"~~~\n{translation}\n~~~")
