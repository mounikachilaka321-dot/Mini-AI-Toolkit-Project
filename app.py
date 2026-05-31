import streamlit as st
from transformers import pipeline

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mini AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Toolkit")
st.write("Sentiment Analysis | Text Generation | Question Answering | Summary Generation")

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_text_generator():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

sentiment = load_sentiment_model()
generator = load_text_generator()
qa = load_qa_model()

# ==========================================
# USER INPUT
# ==========================================

text = st.text_area(
    "Enter a paragraph",
    height=200
)

# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

if st.button("Run AI Toolkit"):

    if not text.strip():
        st.warning("Please enter a paragraph.")
        st.stop()

    # --------------------------------------
    # 1. SENTIMENT ANALYSIS
    # --------------------------------------

    st.subheader("1️⃣ Sentiment Analysis")

    sentiment_result = sentiment(text)

    st.write("**Label:**", sentiment_result[0]["label"])
    st.write("**Confidence:**", round(sentiment_result[0]["score"], 4))

    # --------------------------------------
    # 2. TEXT GENERATION
    # --------------------------------------

    st.subheader("2️⃣ Text Generation")

    generated = generator(
        text,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8
    )

    st.write(generated[0]["generated_text"])

    # --------------------------------------
    # 3. QUESTION ANSWERING
    # --------------------------------------

    st.subheader("3️⃣ Question Answering")

    question = st.text_input(
        "Ask a question based on the paragraph"
    )

    if question:

        answer = qa(
            question=question,
            context=text
        )

        st.write("**Answer:**", answer["answer"])
        st.write("**Score:**", round(answer["score"], 4))

    # --------------------------------------
    # 4. SUMMARY GENERATION
    # --------------------------------------

    st.subheader("4️⃣ Summary Generation")

    summary_prompt = f"""
    Summarize the following paragraph in 2 lines:

    {text}

    Summary:
    """

    summary = generator(
        summary_prompt,
        max_new_tokens=40,
        do_sample=False
    )

    summary_text = summary[0]["generated_text"]

    summary_text = summary_text.replace(
        summary_prompt,
        ""
    )

    st.write(summary_text)

    st.success("Mini AI Toolkit Execution Completed!")