# Kidney Disease Detection - FastAPI Backend
# Save as: main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os
import json
#import openai  # Optional: for chatbot

# ==================== CONFIGURATION ====================
app = FastAPI(title="Kidney Disease Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "kidney_disease_model.pth"
CLASS_NAMES = ['cyst', 'normal', 'stone', 'tumor']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DB_PATH = "kidney_detection.db"
UPLOAD_DIR = "uploads"

# OpenAI API Key (optional - for chatbot)
# openai.api_key = "your-api-key-here"  # Uncomment and add your key

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== DATABASE SETUP ====================
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_path TEXT,
            prediction_class TEXT,
            confidence_score REAL,
            all_probabilities TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Chat history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            message TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_database()

# ==================== MODEL LOADING ====================
model = None
transform = None

def load_model():
    """Load the trained Vision Transformer model"""
    global model, transform
    
    try:
        # Load model
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# ==================== PYDANTIC MODELS ====================
class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    contact: str

class ChatMessage(BaseModel):
    patient_id: int
    message: str
    context: dict = {}

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: dict
    image_path: str

# ==================== HELPER FUNCTIONS ====================
def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def predict_image(image_tensor):
    """Make prediction on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        all_probs = {
            CLASS_NAMES[i]: float(probabilities[i].item()) 
            for i in range(len(CLASS_NAMES))
        }
        
        return predicted_class, confidence_score, all_probs

def get_medical_context(prediction, confidence):
    """Generate medical context for predictions"""
    contexts = {
        'normal': "The kidney appears healthy with no abnormalities detected.",
        'stone': "Kidney stones detected. These are hard deposits of minerals and salts. Recommend hydration and medical consultation.",
        'cyst': "Kidney cyst detected. These are fluid-filled sacs. Most are benign but should be monitored by a healthcare provider.",
        'tumor': "Kidney mass detected. Immediate medical consultation is strongly recommended for further evaluation."
    }
    return contexts.get(prediction, "Medical evaluation recommended.")

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "Kidney Disease Detection API", "status": "running"}

@app.post("/api/patients/create")
async def create_patient(patient: PatientCreate):
    """Create a new patient record"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (name, age, gender, contact)
            VALUES (?, ?, ?, ?)
        ''', (patient.name, patient.age, patient.gender, patient.contact))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {"success": True, "patient_id": patient_id, "message": "Patient created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: int):
    """Get patient details and history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get patient info
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = cursor.fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get prediction history
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC
        ''', (patient_id,))
        predictions = cursor.fetchall()
        
        conn.close()
        
        return {
            "patient": dict(patient),
            "predictions": [dict(p) for p in predictions]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients")
async def list_patients():
    """List all patients"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        patients = cursor.fetchall()
        conn.close()
        
        return {"patients": [dict(p) for p in patients]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: int = Form(None)
):
    """Predict kidney disease from CT scan"""
    try:
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read and process image
        image_bytes = await file.read()
        image_tensor, original_image = preprocess_image(image_bytes)
        
        # Make prediction
        predicted_class, confidence, all_probs = predict_image(image_tensor)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        original_image.save(image_path)
        
        # Save to database if patient_id provided
        if patient_id:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (patient_id, image_path, prediction_class, confidence_score, all_probabilities)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, image_path, predicted_class, confidence, json.dumps(all_probs)))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
        else:
            prediction_id = None
        
        # Get medical context
        medical_context = get_medical_context(predicted_class, confidence)
        
        return {
            "success": True,
            "prediction_id": prediction_id,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {k: round(v * 100, 2) for k, v in all_probs.items()},
            "image_path": image_path,
            "medical_context": medical_context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Chatbot endpoint for medical queries"""
    try:
        # Simple rule-based chatbot (replace with OpenAI API if available)
        response = generate_chatbot_response(message.message, message.context)
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (patient_id, message, response)
            VALUES (?, ?, ?)
        ''', (message.patient_id, message.message, response))
        
        conn.commit()
        conn.close()
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{patient_id}")
async def get_chat_history(patient_id: int):
    """Get chat history for a patient"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM chat_history 
            WHERE patient_id = ? 
            ORDER BY timestamp ASC
        ''', (patient_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        return {"history": [dict(h) for h in history]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHATBOT LOGIC ====================
def generate_chatbot_response(user_message, context):
    """Generate chatbot response (simple rule-based or OpenAI)"""
    
    user_message_lower = user_message.lower()
    
    # Rule-based responses
    if any(word in user_message_lower for word in ['stone', 'kidney stone']):
        return ("Kidney stones are hard deposits made of minerals and salts. "
                "Treatment options include drinking plenty of water (2-3 liters/day), "
                "pain medication, and in some cases, medical procedures. "
                "Please consult with a urologist for personalized treatment.")
    
    elif any(word in user_message_lower for word in ['cyst', 'kidney cyst']):
        return ("Kidney cysts are fluid-filled sacs that form in or on the kidneys. "
                "Most kidney cysts are benign (non-cancerous) and don't cause symptoms. "
                "Regular monitoring through imaging tests is usually recommended. "
                "Consult your doctor for specific advice.")
    
    elif any(word in user_message_lower for word in ['tumor', 'cancer', 'mass']):
        return ("A kidney mass requires immediate medical attention. "
                "Please schedule an appointment with a nephrologist or urologist as soon as possible. "
                "They may recommend additional tests like CT scans, MRI, or biopsy to determine the nature of the mass.")
    
    elif any(word in user_message_lower for word in ['normal', 'healthy']):
        return ("Your kidney appears normal in the scan. To maintain kidney health: "
                "drink adequate water, maintain a healthy diet low in sodium, "
                "exercise regularly, and avoid smoking. Regular check-ups are recommended.")
    
    elif any(word in user_message_lower for word in ['diet', 'food', 'eat']):
        return ("For kidney health: Limit sodium intake, eat more fruits and vegetables, "
                "reduce protein if advised by doctor, limit phosphorus and potassium if needed, "
                "and stay hydrated. Avoid processed foods and excessive salt.")
    
    elif any(word in user_message_lower for word in ['treatment', 'cure', 'medicine']):
        return ("Treatment depends on the specific condition. Always consult with a healthcare provider "
                "for personalized treatment plans. This system provides detection only, "
                "not treatment recommendations. Please see a doctor for proper medical advice.")
    
    else:
        return ("I'm here to help answer questions about kidney health. "
                "However, for specific medical advice, please consult with a qualified healthcare provider. "
                "This system is designed for detection purposes and should not replace professional medical consultation.")

# ==================== OPTIONAL: OpenAI Integration ====================
def generate_chatbot_response_openai(user_message, context):
    """Alternative: Use OpenAI API for chatbot (requires API key)"""
    try:
        system_prompt = """You are a medical assistant specialized in kidney health. 
        Provide helpful information about kidney diseases, but always remind users to 
        consult with healthcare professionals for medical advice. Be empathetic and clear."""
        
        if 'prediction' in context:
            system_prompt += f"\n\nContext: The patient's recent scan shows: {context['prediction']}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except:
        return generate_chatbot_response(user_message, context)

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)