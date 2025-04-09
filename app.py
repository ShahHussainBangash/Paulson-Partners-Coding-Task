import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Set up Streamlit page
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="🧠")
st.title("🧠 Emotion-Aware Conversational AI")

# Load model + tokenizer
@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

# Emotion-based tone responses + emojis
tone_responses = {
    "sadness": "😢 I'm here for you. Want to talk about it?",
    "joy": "😊 That's awesome! Tell me more!",
    "love": "❤️ Aww, that's sweet!",
    "anger": "😡 I get that. Want to let it out?",
    "fear": "😨 It's okay to be scared. You're not alone.",
    "surprise": "😲 Wow! That’s unexpected. What happened?"
}

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("You:", placeholder="Type your message here...")

# Process input
if user_input:
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    # Predict emotion
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        emotion = id2label[pred_class]
        confidence = probs[0][pred_class].item()

    # Generate bot response
    response = tone_responses.get(emotion, "🤖 I'm listening...")

    # Store in history
    st.session_state.chat_history.append({
        "user": user_input,
        "emotion": emotion,


        "confidence": confidence,
        "response": response
    })

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for chat in reversed(st.session_state.chat_history):  # Show newest at top
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Detected Emotion:** `{chat['emotion']}` {tone_responses.get(chat['emotion'], '')[:2]} | **Confidence:** `{chat['confidence']:.2f}`")
        st.markdown(f"**Bot:** {chat['response']}")
        st.markdown("---")
