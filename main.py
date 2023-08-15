import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from keybert import KeyBERT
from bertopic import BERTopic

load_dotenv()
kw_model = os.getenv("KEYWORD_MODEL")
topic_model = os.getenv("TOPIC_MODEL")

kw = KeyBERT(model=kw_model)
topic = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
app = Flask(__name__)

@app.route("/analyze", methods=["GET"])
def index():
    args = request.args
    text = args.get("text")
    if text is None or type(text) != str:
        return "No text or text is not a string", 400
    
    keywords = kw.extract_keywords(text)
    topics, probs = topic.transform([text])

if __name__ == "__main__":
    app.run(debug=True)

    

