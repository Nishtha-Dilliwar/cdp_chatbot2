from flask import Flask, request, jsonify, render_template
import requests
import os
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Hugging Face API configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Your Hugging Face token
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to store extracted text
uploaded_text = ""

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to ask Hugging Face model
def ask_hf(question, context=""):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"Use this context to answer the question: {context}\n\nQuestion: {question}",
        "parameters": {"max_length": 200}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    global uploaded_text
    # Get user input from the request
    user_input = request.json.get('message')
    
    # Ask Hugging Face model for a response with uploaded text as context
    response = ask_hf(user_input, uploaded_text)
    
    # Return the response as JSON
    return jsonify({"response": response})

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_text
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract text from the file
    if filename.endswith('.pdf'):
        uploaded_text = extract_text_from_pdf(file_path)
    else:
        uploaded_text = file.read().decode('utf-8')
    
    # Return the extracted text
    return jsonify({
        "filename": filename,
        "text": uploaded_text,
        "message": "File uploaded and text extracted successfully."
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)