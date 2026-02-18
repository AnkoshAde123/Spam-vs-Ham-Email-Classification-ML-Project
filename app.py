import streamlit as st
import pandas as pd
import pickle


# -------------------------------
# Load pipeline model
# -------------------------------
with open("model/spamham.pkl", "rb") as f:
    model = pickle.load(f)



# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Problem Statement",
                                  "Dataset Overview",
                                  "Analysis",
                                  "Prediction"])


# ===========================================
# PAGE 1 - Problem Statement
# ===========================================
if page == "Problem Statement":
    st.title("📌 SMS Spam Classification Project")

    st.write("""
    ### Objective  
    Build an NLP model that classifies SMS messages as **Ham (Normal)** or **Spam (Fraud message)**.

    ### Why?
    - Spam SMS causes fraud, cheating, phishing, scams, money loss.
    - Telecom companies must detect these texts automatically.
    - Machine Learning helps us classify messages in real-time.

    ### Dataset Source
    - Dataset: `spam.csv`
    - ~5K SMS messages
    - Labels: **ham**, **spam**
    """)

    st.info("Use the Sidebar menu to explore more →")



# ===========================================
# PAGE 2 - Dataset Overview
# ===========================================
elif page == "Dataset Overview":
    st.title("📊 Dataset Preview")

    df = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'text']

    st.write("### Sample Rows")
    st.dataframe(df.head())

    st.write("### Shape")
    st.write(df.shape)

    st.write("### Label Distribution")
    st.bar_chart(df['label'].value_counts())



# ===========================================
# PAGE 3 - Analysis & Insights
# ===========================================
elif page == "Analysis":
    st.title("📈 Exploratory Data Analysis")

    df = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'text']

    st.write("### Total Messages")
    st.write(len(df))

    st.write("### Spam vs Ham distribution")
    st.bar_chart(df['label'].value_counts())

    st.write("### Word Count Distribution")
    df["words"] = df['text'].apply(lambda x: len(str(x).split()))

    st.line_chart(df['words'])

    st.write("""
    ### Key Findings  
    - Spam messages are longer and contain more promotional wording  
    - Many URLs & prize-related words occur in spam  
    - Ham SMS are informal conversation type  
    """)



# ===========================================
# PAGE 4 - Prediction Page
# ===========================================
elif page == "Prediction":
    st.title("🧪 SMS Spam Prediction")

    text = st.text_area("Enter SMS text")

    if st.button("Predict"):
        df_input = pd.DataFrame({"text": [text]})
        result = model.predict(df_input)[0]

        if result == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✔ This is a HAM (safe) message.")


    st.caption("Model loaded from models/spamham.pkl")
