import streamlit as st
import random 
import json 
import os
import nltk 
from nltk.tokenize  import word_tokenize 
from nltk.stem import WordNetLemmatizer 
# ML
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
emotion_model = pipeline( 
 "text-classification", 
 model="j-hartmann/emotion-english-distilroberta-base" ) 
lemmatizer = WordNetLemmatizer()
MEMORY_FILE = "memory.json"
def load_memory():
   if os.path.exists(MEMORY_FILE): 
    try: 
        with open(MEMORY_FILE, "r") as f:
             return json.load(f) 
    except: 
             return {} 
    return {} 
def save_memory(data): 
   with open(MEMORY_FILE, "w") as f: 
      json.dump(data, f)
memory = load_memory()
# Load dataset 
with open("data.txt", "r") as f: 
    data = f.read().split("\n") # NLP preprocess 
def preprocess(text): 
    tokens = word_tokenize(text.lower()) 
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word.isalnum()])
# Build TF-IDF Model
vectorizer = TfidfVectorizer() 
X = vectorizer.fit_transform(data)
def chatbot_response(user_input):
    emotion = emotion_model(user_input)[0]['label'] 
    # Memory feature 
    if user_input.lower() in ["hi", "hello", "hey"]: 
        return random.choice(["Hello!", "Hi there!", "Hey! How can I help you?"]) 
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1].strip().title()
        memory["name"] = name
        save_memory(memory)
        return f"Nice to meet you, {name}!"
    if "what is my name" in user_input.lower(): 
        return f"Your name is {memory.get('name', 'not stored yet')}"
    user_input_processed = preprocess(user_input) 
    user_vec = vectorizer.transform([user_input_processed]) 
    similarity = cosine_similarity(user_vec, X) 
    index = similarity.argmax() 
    emotion = emotion_model(user_input)[0]['label']
    text = user_input.lower()
    if any(word in text for word in ["help", "assist", "support"]):  
        return "Of course! I'm here to assist you 😊"  
    result = emotion_model(user_input)[0] 
    emotion = result['label'] 
    score = result['score'] 
    if score > 0.7: 
        if emotion == "sadness": 
            return "I'm here for you 💙" 
        elif emotion == "fear": 
            return "That sounds worrying. You're not alone." 
        elif emotion == "anger": 
            return "I understand you're upset." 
        elif emotion== "happiness": 
            return " That's great to hear!😄"
        processed = preprocess(user_input) 
        user_vec = vectorizer.transform([processed]) 
        similarity = cosine_similarity(user_vec, X) 
        index = similarity.argmax() 
        return data[index]
st.markdown(""" 
<style> 
body {
    background-color: white;
} 
.chat-container { 
    display: flex; 
    flex-direction: column; 
} 
/* USER MESSAGE (RIGHT) */ 
.user-msg { 
    align-self: flex-end; 
    background-color: #0078FF; 
    color: white; 
    padding: 10px; 
    margin: 5px; 
    border-radius: 10px; 
    max-width: 60%; 
    position: relative;
 } 
            /* RIGHT ARROW */ 
.user-msg::after { 
    content: ''; 
    position: absolute; 
    right: -10px;
     top: 10px; 
    border-width: 8px; border-style: solid; 
    border-color: transparent transparent transparent #0078FF; }
 /* BOT MESSAGE (LEFT) */ 
.bot-msg { 
    align-self: flex-start; 
    background-color: #0078FF;
     color: white; padding: 10px; 
    margin: 5px; border-radius: 10px; 
    max-width: 60%; position: relative; }
 /* LEFT ARROW */ 
.bot-msg::after { 
    content: ''; 
    position: absolute; left: -10px; 
    top: 10px; border-width: 8px; 
    border-style: solid; 
    border-color: transparent #0078FF transparent transparent; }
    </style> 
    """, unsafe_allow_html=True) 
# ---------------- UI ---------------- # 
st.title("🤖 HELPMATE") 
if "chat" not in st.session_state: 
    st.session_state.chat = [] 
    # Display chat 
st.markdown('<div class="chat-container">', 
            unsafe_allow_html=True) 

for sender, msg in st.session_state.chat: 
    if sender == "user": 
        st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True) 
    else: 
        st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True) 
        st.markdown('</div>', unsafe_allow_html=True) 
        # Input 
user_input = st.chat_input("Type your message...") 
if user_input: 
    st.session_state.chat.append(("user", user_input)) 
    reply = chatbot_response(user_input) 
    st.session_state.chat.append(("assistant", reply))
st.rerun()   