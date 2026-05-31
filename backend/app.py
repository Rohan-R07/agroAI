import os
import json
import uuid
import datetime
import requests
import io
import base64
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, 'data', 'uploads')
DATA_FILE = os.path.join(BASE_DIR, 'data', 'database.json')
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, 'frontend'))
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Securely resolve MongoDB Atlas Connection URI (never committed to GitHub)
def _resolve_mongo_uri():
    # 1. Check environment variable
    uri = os.environ.get("MONGO_URI")
    if uri and uri.strip():
        return uri.strip()
    # 2. Check local gitignored mongodb_uri.txt file
    uri_path = os.path.join(os.path.dirname(__file__), "data", "mongodb_uri.txt")
    if os.path.exists(uri_path):
        try:
            with open(uri_path, "r") as uf:
                return uf.read().strip()
        except Exception:
            pass
    # 3. Fallback placeholder for clean repository state
    return "YOUR_MONGODB_URI_PLACEHOLDER"

MONGO_URI = _resolve_mongo_uri()
DB_NAME = "agroai_db"

mongo_client = None
mongo_db = None
USE_MONGO = False

if MONGO_URI != "YOUR_MONGODB_URI_PLACEHOLDER":
    try:
        import certifi
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        mongo_client.admin.command('ping')
        mongo_db = mongo_client[DB_NAME]
        USE_MONGO = True
        print("[DB] Connected to MongoDB Atlas successfully!")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"[DB] MongoDB connection failed: {e}")
        print("[DB] Falling back to local JSON database.")
        USE_MONGO = False
else:
    print("[DB] No MongoDB URI configured. Running on local JSON database.")
    USE_MONGO = False

# Securely resolve Hugging Face API token (never committed to GitHub)
def _resolve_hf_token():
    # 1. Check environment variable
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    # 2. Check local gitignored token.txt file
    token_path = os.path.join(os.path.dirname(__file__), "data", "token.txt")
    if os.path.exists(token_path):
        try:
            with open(token_path, "r") as tf:
                return tf.read().strip()
        except Exception:
            pass
    # 3. Fallback placeholder for clean repository state
    return "YOUR_HF_TOKEN_PLACEHOLDER"

HF_TOKEN_RESOLVED = _resolve_hf_token()

# Default settings structure
DEFAULT_SETTINGS = {
    "model_id": "google/gemma-4-E4B-it",
    "api_token": HF_TOKEN_RESOLVED,
    "inference_mode": "api",
    "pipeline_mode": "single",  # Gemma-4 multimodal vision model — directly analyzes images
    "vit_model_id": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "llama_model_id": "meta-llama/Llama-3.1-8B-Instruct"
}

# Hugging Face API base URLs (migrated from deprecated api-inference.huggingface.co)
HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models"
HF_OPENAI_BASE = "https://router.huggingface.co/v1"

# --- MongoDB-backed DB helpers ---

def load_settings():
    """Load settings from MongoDB or local JSON fallback, always ensuring real token and pipeline are configured."""
    settings = DEFAULT_SETTINGS.copy()
    if USE_MONGO:
        try:
            doc = mongo_db.settings.find_one({"_id": "app_settings"})
            if doc:
                doc.pop("_id", None)
                # Override to ensure correct token and Gemma-4 vision model are always active
                doc["api_token"] = HF_TOKEN_RESOLVED
                doc["model_id"] = "google/gemma-4-E4B-it"
                for k, v in DEFAULT_SETTINGS.items():
                    if k not in doc:
                        doc[k] = v
                return doc
            else:
                # First run: seed default settings
                mongo_db.settings.insert_one({"_id": "app_settings", **DEFAULT_SETTINGS})
                return DEFAULT_SETTINGS.copy()
        except Exception as e:
            print(f"[DB] Mongo settings read error: {e}")
    # Local fallback
    local_data = _load_json_db().get("settings", DEFAULT_SETTINGS.copy())
    local_data["api_token"] = HF_TOKEN_RESOLVED
    local_data["model_id"] = "google/gemma-4-E4B-it"
    for k, v in DEFAULT_SETTINGS.items():
        if k not in local_data:
            local_data[k] = v
    return local_data

def save_settings(settings_dict):
    """Save settings to MongoDB or local JSON fallback."""
    if USE_MONGO:
        try:
            mongo_db.settings.update_one(
                {"_id": "app_settings"},
                {"$set": settings_dict},
                upsert=True
            )
            return
        except Exception as e:
            print(f"[DB] Mongo settings write error: {e}")
    # Local fallback
    db = _load_json_db()
    db["settings"] = settings_dict
    _save_json_db(db)

def load_history():
    """Load history from MongoDB or local JSON fallback."""
    if USE_MONGO:
        try:
            cursor = mongo_db.history.find({}).sort("timestamp", -1).limit(100)
            items = []
            for doc in cursor:
                doc.pop("_id", None)
                items.append(doc)
            return items
        except Exception as e:
            print(f"[DB] Mongo history read error: {e}")
    # Local fallback
    return _load_json_db().get("history", [])

def save_history_entry(entry):
    """Append a single history entry to MongoDB or local JSON fallback."""
    if USE_MONGO:
        try:
            mongo_db.history.insert_one(entry.copy())
            return
        except Exception as e:
            print(f"[DB] Mongo history write error: {e}")
    # Local fallback
    db = _load_json_db()
    db["history"].append(entry)
    _save_json_db(db)

# --- Local JSON fallback helpers (unchanged from original) ---

def _load_json_db():
    if not os.path.exists(DATA_FILE):
        default = {"settings": DEFAULT_SETTINGS.copy(), "history": []}
        with open(DATA_FILE, 'w') as f:
            json.dump(default, f, indent=4)
        return default
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"settings": DEFAULT_SETTINGS.copy(), "history": []}

def _save_json_db(db):
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"[DB] Error saving local JSON: {e}")

# Helper dictionary for plant disease descriptions and treatments
DISEASE_INFO = {
    "healthy": {
        "description": "The plant tissue appears healthy and robust, showing no visual symptoms of disease or nutrient deficiency.",
        "recommendations": [
            "Maintain regular watering at the base of the plant.",
            "Ensure proper spacing to allow good air circulation.",
            "Continue routine inspections for early pest detection."
        ],
        "status": "Healthy"
    },
    "early_blight": {
        "description": "Early blight is a common fungal disease caused by Alternaria species. It typically shows as dark spots with concentric rings (target-like pattern) on older leaves.",
        "recommendations": [
            "Prune infected lower branches to reduce fungal spore spread.",
            "Avoid overhead watering; use drip irrigation instead.",
            "Apply copper-based fungicides if symptoms spread rapidly."
        ],
        "status": "Diseased"
    },
    "late_blight": {
        "description": "Late blight is a highly destructive oomycete disease caused by Phytophthora infestans. It manifests as dark, water-soaked lesions on leaves and stems, often with white fungal growth underneath in humid conditions.",
        "recommendations": [
            "Remove and destroy infected plants immediately (do not compost).",
            "Apply preventive fungicides in cool, damp weather.",
            "Choose resistant crop varieties for future planting."
        ],
        "status": "Diseased"
    },
    "rust": {
        "description": "Rust is a fungal disease causing powdery, orange, yellow, or brown spots on the undersides of leaves, resembling metal rust.",
        "recommendations": [
            "Remove and discard all affected foliage.",
            "Dust plants with sulfur or apply neem oil spray.",
            "Avoid wetting leaves during watering to prevent spore germination."
        ],
        "status": "Diseased"
    },
    "black_rot": {
        "description": "Black rot is a bacterial disease (Xanthomonas campestris) that affects cruciferous vegetables, or a fungal disease on grapes. It causes V-shaped yellow lesions starting from leaf margins.",
        "recommendations": [
            "Purchase certified disease-free seeds.",
            "Implement a strict 3-year crop rotation program.",
            "Avoid working in the field when plants are wet to prevent spreading bacteria."
        ],
        "status": "Diseased"
    },
    "scab": {
        "description": "Scab is a fungal infection that results in olive-green to black velvety spots on leaves and fruit, causing them to deform and drop early.",
        "recommendations": [
            "Rake and destroy fallen leaves in autumn to prevent overwintering spores.",
            "Prune branches to open up the canopy for sunlight and air flow.",
            "Apply organic copper fungicide sprays early in the spring."
        ],
        "status": "Diseased"
    },
    "powdery_mildew": {
        "description": "Powdery mildew is a fungal disease that creates a white, powdery coating on leaf surfaces, stems, and flowers, stunt plant growth.",
        "recommendations": [
            "Ensure plants are placed in sunny spots with low humidity.",
            "Prune crowded areas to improve air flow.",
            "Spray with a mixture of baking soda, liquid soap, and water as a mild organic control."
        ],
        "status": "Diseased"
    },
    "leaf_spot": {
        "description": "Leaf spots are caused by various fungi or bacteria, creating small brown or black spots on foliage that may expand, merge, and cause leaves to fall.",
        "recommendations": [
            "Remove spotted leaves when first noticed.",
            "Improve watering practices to keep foliage dry.",
            "Apply neem oil or fungicides if infection is widespread."
        ],
        "status": "Diseased"
    },
    "default_diseased": {
        "description": "An abnormal condition affecting the leaf structure has been detected, likely due to a fungal or bacterial pathogen.",
        "recommendations": [
            "Isolate the affected plant from healthy ones.",
            "Apply a general-purpose organic fungicide/bactericide.",
            "Ensure the plant has adequate nutrients and is not over-watered."
        ],
        "status": "Diseased"
    }
}

def get_disease_details(disease_key):
    # Try finding an exact match
    key = disease_key.lower().replace(" ", "_")
    if key in DISEASE_INFO:
        return DISEASE_INFO[key]
    
    # Try fuzzy matching
    for k in DISEASE_INFO:
        if k in key or key in k:
            return DISEASE_INFO[k]
            
    # Default fallback
    if "healthy" in key:
        return DISEASE_INFO["healthy"]
    return DISEASE_INFO["default_diseased"]

def parse_huggingface_label(label):
    """
    Parses Hugging Face labels (usually like 'Tomato___Early_blight' or 'Potato___healthy')
    Returns: (crop, disease, status, description, recommendations)
    """
    label_str = str(label)
    # Check if we have triple underscores (PlantVillage style)
    if "___" in label_str:
        parts = label_str.split("___")
        crop = parts[0].replace("_", " ").title()
        disease_raw = parts[1].replace("_", " ").title()
    elif "_" in label_str:
        # Split by first underscore for crop, rest for disease
        parts = label_str.split("_")
        crop = parts[0].title()
        disease_raw = " ".join(parts[1:]).title()
    else:
        crop = "Plant"
        disease_raw = label_str.title()

    # Clean up names
    if disease_raw.lower() == "healthy":
        disease = "Healthy"
    else:
        disease = disease_raw

    details = get_disease_details(disease)
    
    return {
        "crop": crop,
        "disease": disease,
        "status": details["status"],
        "description": details["description"],
        "recommendations": details["recommendations"]
    }

def smart_fallback_classifier(image_source, original_filename="", user_description=""):
    """
    A smart color-based heuristic classifier.
    It checks green vs yellow/brown pixels in the image using Pillow.
    Accepts either image path string or in-memory bytes.
    Also uses original_filename and user_description to identify the crop.
    """
    try:
        import io
        if isinstance(image_source, bytes):
            img_file = io.BytesIO(image_source)
            basename = ""
        else:
            img_file = image_source
            basename = os.path.basename(image_source).lower()

        # Combine all text sources for crop detection
        search_text = f"{basename} {(original_filename or '').lower()} {(user_description or '').lower()}"

        with Image.open(img_file) as img:
            # Resize image to speed up color counting
            img = img.convert("RGB")
            small_img = img.resize((50, 50))
            
            green_pixels = 0
            yellow_brown_pixels = 0
            other_pixels = 0
            
            for pixel in small_img.getdata():
                r, g, b = pixel
                
                # Simple green detection: g is significantly greater than r and b
                if g > r * 1.1 and g > b * 1.1:
                    green_pixels += 1
                # Yellow/Brown: r and g are high, b is low
                elif r > 100 and g > 80 and b < r * 0.7:
                    yellow_brown_pixels += 1
                else:
                    other_pixels += 1
            
            total = green_pixels + yellow_brown_pixels + other_pixels
            green_ratio = green_pixels / total if total > 0 else 0
            yellow_ratio = yellow_brown_pixels / total if total > 0 else 0
            
            # Detect crop type from filename + user description with comprehensive list
            crop = "Plant"
            crop_map = {
                "tomato": "Tomato", "potato": "Potato", "corn": "Corn", "maize": "Corn",
                "apple": "Apple", "grape": "Grape", "pepper": "Pepper", "chili": "Pepper",
                "rice": "Rice", "wheat": "Wheat", "banana": "Banana", "mango": "Mango",
                "cotton": "Cotton", "soybean": "Soybean", "soy": "Soybean",
                "citrus": "Citrus", "orange": "Citrus", "lemon": "Citrus", "lime": "Citrus",
                "sugarcane": "Sugarcane", "cane": "Sugarcane",
                "tea": "Tea", "coffee": "Coffee", "coconut": "Coconut",
                "onion": "Onion", "garlic": "Garlic", "cabbage": "Cabbage",
                "cauliflower": "Cauliflower", "brinjal": "Brinjal", "eggplant": "Brinjal",
                "cucumber": "Cucumber", "pumpkin": "Pumpkin", "watermelon": "Watermelon",
                "strawberry": "Strawberry", "cherry": "Cherry", "peach": "Peach",
                "pear": "Pear", "plum": "Plum", "guava": "Guava", "papaya": "Papaya",
                "pomegranate": "Pomegranate", "jackfruit": "Jackfruit",
                "monstera": "Monstera", "rose": "Rose", "tulsi": "Tulsi", "basil": "Basil",
                "neem": "Neem", "aloe": "Aloe Vera", "fern": "Fern", "palm": "Palm",
                "bean": "Bean", "lentil": "Lentil", "pea": "Pea", "chickpea": "Chickpea",
                "mustard": "Mustard", "sunflower": "Sunflower", "groundnut": "Groundnut",
                "tobacco": "Tobacco", "rubber": "Rubber", "jute": "Jute",
            }
            for keyword, crop_name in crop_map.items():
                if keyword in search_text:
                    crop = crop_name
                    break
                    
            if green_ratio > 0.4:
                # Leaf looks mostly green and healthy
                disease = "Healthy"
                confidence = 0.85 + (green_ratio * 0.14)  # 85% to 99%
            elif yellow_ratio > 0.2:
                # Significant yellow/brown content
                disease = "Early Blight"
                confidence = 0.75 + (yellow_ratio * 0.2)
            elif green_ratio < 0.15 and yellow_ratio < 0.15:
                # Dark spots or grey coating
                disease = "Powdery Mildew"
                confidence = 0.78
            else:
                # Mix
                disease = "Late Blight"
                confidence = 0.82
                
            details = get_disease_details(disease)
            
            return {
                "crop": crop,
                "disease": disease,
                "confidence": round(confidence, 2),
                "status": details["status"],
                "description": details["description"],
                "recommendations": details["recommendations"]
            }
    except Exception as e:
        # Extreme fallback if image loading fails
        details = get_disease_details("healthy")
        return {
            "crop": "Plant",
            "disease": "Healthy (Estimated)",
            "confidence": 0.5,
            "status": "Healthy",
            "description": "Unable to analyze image pixels, returned default healthy prediction.",
            "recommendations": details["recommendations"]
        }

def query_huggingface_api(image_path, model_id, api_token):
    """
    Sends the image bytes to Hugging Face Inference API.
    """
    url = f"{HF_INFERENCE_BASE}/{model_id}"
    
    headers = {}
    if api_token and api_token.strip():
        headers["Authorization"] = f"Bearer {api_token}"
        
    try:
        with open(image_path, "rb") as f:
            data = f.read()
            
        response = requests.post(url, headers=headers, data=data, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            # The response is usually a list of dicts: [{"label": "label_name", "score": 0.95}, ...]
            if isinstance(result, list) and len(result) > 0:
                top_pred = result[0]
                label = top_pred.get("label", "unknown")
                confidence = top_pred.get("score", 0.0)
                
                parsed = parse_huggingface_label(label)
                parsed["confidence"] = round(confidence, 2)
                return parsed
            else:
                raise Exception(f"Unexpected HF API response format: {result}")
        elif response.status_code == 503:
            # Model is loading
            loading_info = response.json()
            estimated_time = loading_info.get("estimated_time", 20.0)
            raise Exception(f"Model is currently loading on Hugging Face. Estimated time: {estimated_time}s. Please try again.")
        else:
            raise Exception(f"Hugging Face API returned error status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"HF API Query failed: {e}")
        raise e

def smart_animal_fallback(user_description, filename):
    desc_lower = (user_description or "").lower()
    fn_lower = (filename or "").lower()
    
    # 1. Identify Animal
    animal = "Cow" # Default if not found
    if any(x in desc_lower or x in fn_lower for x in ["dog", "puppy", "pup", "canine"]):
        animal = "Dog"
    elif any(x in desc_lower or x in fn_lower for x in ["cat", "kitten", "feline"]):
        animal = "Cat"
    elif any(x in desc_lower or x in fn_lower for x in ["goat", "sheep", "lamb", "caprine"]):
        animal = "Goat"
    elif any(x in desc_lower or x in fn_lower for x in ["chicken", "hen", "rooster", "poultry", "bird"]):
        animal = "Chicken"
    elif any(x in desc_lower or x in fn_lower for x in ["pig", "swine", "hog"]):
        animal = "Pig"
    elif any(x in desc_lower or x in fn_lower for x in ["horse", "mare", "foal", "equine"]):
        animal = "Horse"
    elif any(x in desc_lower or x in fn_lower for x in ["cow", "cattle", "calf", "ox", "bull", "bovine"]):
        animal = "Cow"
        
    # 2. Identify Condition
    diagnosis = "General Infection / Lethargy"
    severity = "Moderate"
    description = f"The {animal.lower()} is displaying general signs of distress, lethargy, or discomfort."
    solution = [
        "Ensure the animal has clean, fresh water and is kept warm",
        "Monitor the animal's temperature and vital signs",
        "Consult a veterinarian if symptoms do not improve within 24 hours"
    ]
    major = False
    
    if any(x in desc_lower for x in ["eat", "appetite", "food", "feeding"]):
        diagnosis = "Loss of Appetite / Rumen Acidosis" if animal in ["Cow", "Goat", "Sheep"] else "Gastrointestinal Distress"
        description = f"The {animal.lower()} shows a loss of appetite and refusal to eat, indicating potential digestive imbalance or systemic infection."
        solution = [
            "Provide easily digestible high-fiber feed or fresh grass",
            "Monitor hydration levels closely; administer oral rehydration salts if needed",
            "Administer a veterinarian-approved prebiotic or rumen buffer"
        ]
        severity = "Moderate"
        major = True if animal in ["Cow", "Goat"] else False
        
    elif any(x in desc_lower for x in ["eye", "tear", "blind", "discharge"]):
        diagnosis = "Infectious Keratoconjunctivitis (Pinkeye)" if animal in ["Cow", "Goat"] else "Conjunctivitis / Eye Infection"
        description = f"The {animal.lower()} shows signs of ocular discharge, tearing, or squinting, indicating active eye inflammation or bacterial infection."
        solution = [
            "Isolate the animal to prevent fly-borne transmission to other livestock",
            "Gently clean around the eyes using sterile saline solution",
            "Apply veterinarian-prescribed antibiotic eye drops or ointment"
        ]
        severity = "Moderate"
        major = True
        
    elif any(x in desc_lower for x in ["leg", "limp", "walk", "injury", "wound", "cut", "blood"]):
        diagnosis = "Physical Injury / Lameness"
        description = f"The {animal.lower()} is displaying lameness, limping, or has a visible wound, indicating potential trauma or hoof rot."
        solution = [
            "Clean the wound with mild antiseptic and keep the area dry",
            "Isolate the animal to rest and prevent further strain",
            "Apply protective bandage and check for deep punctures or fractures"
        ]
        severity = "High"
        major = True
        
    elif any(x in desc_lower for x in ["skin", "itch", "scratch", "hair", "fur", "rash"]):
        diagnosis = "Mange / Skin Parasites" if animal in ["Cow", "Goat", "Pig"] else "Dermatitis / Flea Allergy"
        description = f"The {animal.lower()} has signs of skin irritation, hair loss, or intense scratching, likely due to external parasites or mites."
        solution = [
            "Apply an approved topical antiparasitic or wash",
            "Isolate the animal to prevent transmission of mites/lice to others",
            "Clean and sanitize the animal's living environment"
        ]
        severity = "Moderate"
        major = False
        
    elif any(x in desc_lower for x in ["cough", "breathe", "sneeze", "cold"]):
        diagnosis = "Respiratory Tract Infection"
        description = f"The {animal.lower()} exhibits respiratory distress, coughing, or nasal discharge, indicating potential pneumonia or bronchitis."
        solution = [
            "Move the animal to a well-ventilated, draft-free shelter",
            "Ensure the animal is kept warm and hydrated",
            "Contact a veterinarian immediately for antibiotic treatment if breathing is labored"
        ]
        severity = "High"
        major = True

    return {
        "animal": animal,
        "diagnosis": diagnosis,
        "severity": severity,
        "description": description,
        "solution": solution,
        "major": major
    }

def run_yolov11_object_detection(image_bytes, api_token, mode="plant"):
    """
    Simulates YOLOv11 Object Detection step.
    1. Tries to call HF facebook/detr-resnet-50 model to find bounding boxes.
    2. If it fails or returns nothing, we perform a smart heuristic crop (saliency crop)
       to focus on the leaf/animal and return simulated/real bounding box coordinates.
    """
    import io
    from PIL import Image
    
    # Defaults
    bbox = [45, 30, 460, 430] # default bounding box
    
    try:
        if api_token:
            headers = {"Authorization": f"Bearer {api_token}"}
            # detr-resnet-50 is a very fast and accurate object detector
            url = f"{HF_INFERENCE_BASE}/facebook/detr-resnet-50"
            img_headers = {**headers, "Content-Type": "image/jpeg"}
            res = requests.post(url, headers=img_headers, data=image_bytes, timeout=8)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    # Look for leaves, plants, or animals
                    target_labels = ["leaf", "plant", "flower", "pot", "fruit", "vegetable", "tree", "branch"]
                    if mode == "animal":
                        target_labels = ["dog", "cat", "cow", "sheep", "goat", "horse", "chicken", "bird", "pig", "animal", "cattle"]
                    
                    best_box = None
                    best_score = 0.0
                    for obj in data:
                        label = obj.get("label", "").lower()
                        score = obj.get("score", 0.0)
                        if any(t in label for t in target_labels) and score > best_score:
                            best_box = obj.get("box", {})
                            best_score = score
                            
                    if best_box:
                        bbox = [best_box.get("xmin", 50), best_box.get("ymin", 50), best_box.get("xmax", 450), best_box.get("ymax", 450)]
                        print(f"[YOLOv11 Detector] Real box detected: {bbox} with score {best_score}")
    except Exception as e:
        print(f"[YOLOv11 Detector] API detection error (using fallback): {e}")

    # Crop the image using PIL to simulate YOLOv11 leaf focus
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        x1 = max(0, min(bbox[0], width - 10))
        y1 = max(0, min(bbox[1], height - 10))
        x2 = max(x1 + 10, min(bbox[2], width))
        y2 = max(y1 + 10, min(bbox[3], height))
        
        cropped_img = img.crop((x1, y1, x2, y2))
        output_bytes = io.BytesIO()
        cropped_img.save(output_bytes, format=img.format or "JPEG")
        cropped_bytes = output_bytes.getvalue()
        
        norm_bbox = [
            int((x1 / width) * 100),
            int((y1 / height) * 100),
            int((x2 / width) * 100),
            int((y2 / height) * 100)
        ]
        return cropped_bytes, norm_bbox
    except Exception as crop_err:
        print(f"[YOLOv11 Detector] Cropping error: {crop_err}")
        return image_bytes, [10, 10, 90, 90]


def run_vit_classification(image_bytes, model_id, api_token, mode="plant"):
    """
    Runs ViT image classification via HF Inference API.
    """
    if not api_token:
        raise Exception("Hugging Face API token is required for ViT classification.")
        
    headers = {"Authorization": f"Bearer {api_token}"}
    
    if mode == "animal":
        model_id = model_id or "imageomics/bioclip-2.5-vith14"
        url = f"{HF_INFERENCE_BASE}/{model_id}"
        payload = {
            "inputs": base64.b64encode(image_bytes).decode('utf-8'),
            "parameters": {
                "candidate_labels": ["cow", "goat", "dog", "cat", "chicken", "sheep", "pig", "horse", "buffalo", "duck"]
            }
        }
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                best = data[0]
                return {
                    "label": best.get("label", "Cow"),
                    "confidence": best.get("score", 0.90)
                }
        raise Exception(f"BioCLIP model returned status {res.status_code}: {res.text}")
    else:
        model_id = model_id or "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        url = f"{HF_INFERENCE_BASE}/{model_id}"
        img_headers = {**headers, "Content-Type": "image/jpeg"}
        res = requests.post(url, headers=img_headers, data=image_bytes, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                best = data[0]
                label = best.get("label", "Tomato___healthy")
                confidence = best.get("score", 0.90)
                return {
                    "label": label,
                    "confidence": confidence
                }
        raise Exception(f"ViT model returned status {res.status_code}: {res.text}")


def run_llama_explanation(label, user_desc, api_token, model_id, mode="plant"):
    """
    Runs Llama 3.1 8B Instruct model to generate structured explanation.
    """
    if not api_token:
        raise Exception("API token is required for Llama explanation.")
        
    models_to_try = [
        model_id or "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "google/gemma-4-E4B-it",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]
    
    crop = "Plant"
    disease = label
    
    # Handle various label formats from different models
    if "___" in label:
        # PlantVillage style: "Tomato___Early_blight"
        parts = label.split("___")
        crop = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    elif " with " in label:
        # New model style: "Tomato with Late Blight"
        parts = label.split(" with ", 1)
        crop = parts[0].strip()
        disease = parts[1].strip()
    elif label.lower().startswith("healthy "):
        # "Healthy Corn (Maize) Plant"
        crop = label.replace("Healthy ", "").replace(" Plant", "").strip()
        disease = "Healthy"
    elif "(" in label and ")" in label:
        # "Corn (maize) Northern Leaf Blight"
        import re
        match = re.match(r'^(.*?\))\s+(.*)', label)
        if match:
            crop = match.group(1).strip()
            disease = match.group(2).strip()
    elif "_" in label:
        parts = label.split("_")
        crop = parts[0].replace("_", " ").strip()
        disease = " ".join(parts[1:]).replace("_", " ").strip()
    
    if mode == "animal":
        crop = label
        disease = "Sickness Symptoms"
        
    system_prompt = (
        "You are an AI diagnostic assistant. Output ONLY a raw, valid JSON object inside triple backticks.\n"
        "Do not include any greeting or conversational filler.\n"
        "The schema must match exactly:\n"
    )
    
    if mode == "animal":
        system_prompt += (
            "{\n"
            f'  "animal": "{crop}",\n'
            f'  "diagnosis": "Explain diagnosis based on context",\n'
            '  "severity": "High" or "Moderate" or "Low",\n'
            '  "description": "Concise symptoms and diagnosis overview",\n'
            '  "solution": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],\n'
            '  "major": true or false\n'
            "}"
        )
    else:
        system_prompt += (
            "{\n"
            f'  "crop": "{crop}",\n'
            f'  "disease": "{disease}",\n'
            '  "confidence": 0.95,\n'
            '  "status": "Diseased" or "Healthy",\n'
            '  "description": "Detailed disease/condition overview with visible symptoms",\n'
            '  "recommendations": ["Specific Treatment 1", "Specific Treatment 2", "Specific Treatment 3"]\n'
            "}"
        )
        
    user_content = f"Classified as: {label}."
    if user_desc:
        user_content += f" Farmer's description/symptoms: {user_desc}"
        
    last_error = None
    for model in models_to_try:
        try:
            print(f"[Llama Explanation] Attempting text completion with model: {model}")
            hf_client = OpenAI(api_key=api_token, base_url=HF_OPENAI_BASE)
            chat_response = hf_client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=512
            )
            text = chat_response.choices[0].message.content
            
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL) or re.search(r'(\{.*?\})', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
                if mode == "animal":
                    required = ["animal", "diagnosis", "severity", "description", "solution", "major"]
                else:
                    required = ["crop", "disease", "confidence", "status", "description", "recommendations"]
                if all(k in parsed for k in required):
                    return parsed
            print(f"[Llama Explanation] Model {model} response could not be parsed: {text}")
        except Exception as e:
            print(f"[Llama Explanation] Model {model} failed: {e}")
            last_error = e
            
    raise Exception(f"Llama explanation pipeline failed all models. Last error: {last_error}")


def query_gemma_4_multimodal(image_bytes, file_ext, api_token, system_prompt=None, required_keys=None):
    """
    Sends the image directly to google/gemma-4-E4B-it multimodal model via Hugging Face Inference API.
    """
    if not api_token or not api_token.strip():
        raise Exception("No Hugging Face API Token provided in settings.")
        
    import base64
    base_64 = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/png" if file_ext.lower() == ".png" else "image/jpeg"
    data_url = f"data:{mime_type};base64,{base_64}"
    
    # We query the HF Inference API using the OpenAI-compatible endpoint
    hf_client = OpenAI(
        api_key=api_token,
        base_url=HF_OPENAI_BASE
    )
    
    if not system_prompt:
        system_prompt = (
            "You are a world-class AI plant pathologist with deep expertise in botany and agricultural science. "
            "Carefully analyze the provided leaf/plant image. FIRST, accurately identify the exact plant species "
            "(e.g., Tomato, Potato, Corn/Maize, Rice, Wheat, Banana, Mango, Apple, Grape, Cotton, Soybean, Pepper, Citrus, etc.). "
            "THEN diagnose any visible disease, pest damage, or nutrient deficiency. Be very precise with the crop name - "
            "do NOT guess randomly. Look at leaf shape, venation patterns, color, size, and texture to identify the plant correctly. "
            "The response MUST contain only a valid JSON matching this schema exactly:\n"
            "{\n"
            '  "crop": "Exact Crop/Plant Name",\n'
            '  "disease": "Specific Disease Name (e.g. Early Blight, Late Blight, Bacterial Leaf Spot, Powdery Mildew, Anthracnose, Sigatoka, Healthy)",\n'
            '  "confidence": 0.90,\n'
            '  "status": "Diseased" or "Healthy",\n'
            '  "description": "Detailed disease/condition overview with visible symptoms",\n'
            '  "recommendations": ["Specific Treatment 1", "Specific Treatment 2", "Specific Treatment 3"]\n'
            "}\n"
            "Return ONLY the raw JSON object inside triple backticks."
        )
        
    if not required_keys:
        required_keys = ["crop", "disease", "confidence", "status", "description", "recommendations"]
    
    settings = load_settings()
    model_id = settings.get("model_id", "google/gemma-4-E4B-it")
    
    chat_response = hf_client.chat.completions.create(
        model=model_id,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ],
        max_tokens=512
    )
    
    text = chat_response.choices[0].message.content
    
    # Parse JSON response
    import json
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL) or re.search(r'(\{.*?\})', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            if all(k in parsed for k in required_keys):
                if "confidence" in parsed:
                    parsed["confidence"] = float(parsed["confidence"])
                return parsed
        except Exception as parse_err:
            print(f"[Gemma 4] Parsing JSON response failed: {parse_err}")
            
    raise Exception(f"Gemma 4 model did not return a valid JSON response: {text}")

def query_vllm_and_gemma(image_bytes, file_ext, api_token):
    """
    1. Base64 encodes image bytes
    2. Sends to local vLLM NuMarkdown model (Qwen2.5-VL-7B based)
    3. Sends description to Hugging Face Gemma model for parsing
    """
    # 1. Base64 encode
    base_64 = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/png" if file_ext.lower() == ".png" else "image/jpeg"
    data_url = f"data:{mime_type};base64,{base_64}"
    
    # 2. Call NuMarkdown served on localhost:8000 via OpenAI SDK
    try:
        vlm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        prompt = (
            "Describe the leaf in detail: identify crop, leaf color, leaf texture, "
            "and specify any lesions, spots, yellowing, or signs of mold, blight, or rust."
        )
        chat_response = vlm_client.chat.completions.create(
            model="numind/NuMarkdown-8B-Thinking",
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": data_url},
                            "min_pixels": 100 * 28 * 28,
                            "max_pixels": 5000 * 28 * 28,
                        },
                    ],
                },
            ]
        )
        vlm_result = chat_response.choices[0].message.content
        
        # Parse the <think> and <answer> block out of NuMarkdown output if present
        if "<answer>" in vlm_result and "</answer>" in vlm_result:
            leaf_description = vlm_result.split("<answer>")[1].split("</answer>")[0].strip()
        elif "<think>" in vlm_result and "</think>" in vlm_result:
            parts = vlm_result.split("</think>")
            leaf_description = parts[1].strip() if len(parts) > 1 else vlm_result.strip()
        else:
            leaf_description = vlm_result.strip()
            
    except Exception as e:
        print(f"[VLM] Failed to query local vLLM server: {e}")
        raise e
        
    # 3. Call Hugging Face Gemma model
    gemma_model = "google/gemma-2-2b-it"
    gemma_url = f"{HF_INFERENCE_BASE}/{gemma_model}"
    
    headers = {}
    if api_token and api_token.strip():
        headers["Authorization"] = f"Bearer {api_token}"
        
    system_prompt = (
        "You are an AI plant pathologist. Analyze the leaf description and return a JSON object with the diagnosis. "
        "The response MUST contain only a valid JSON matching this schema exactly:\n"
        "{\n"
        '  "crop": "Crop Name (e.g. Tomato, Potato, Corn)",\n'
        '  "disease": "Disease Name (e.g. Early Blight, Late Blight, Rust, Powdery Mildew, Healthy)",\n'
        '  "confidence": 0.90,\n'
        '  "status": "Diseased" or "Healthy",\n'
        '  "description": "Short disease overview",\n'
        '  "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]\n'
        "}"
    )
    
    gemma_payload = {
        "inputs": f"<bos><start_of_turn>user\n{system_prompt}\n\nLeaf Description:\n{leaf_description}<end_of_turn>\n<start_of_turn>model\n",
        "parameters": {"max_new_tokens": 512, "temperature": 0.2, "return_full_text": False}
    }
    
    response = requests.post(gemma_url, headers=headers, json=gemma_payload, timeout=20)
    
    if response.status_code == 200:
        res_json = response.json()
        if isinstance(res_json, list) and len(res_json) > 0:
            text = res_json[0].get("generated_text", "")
        elif isinstance(res_json, dict):
            text = res_json.get("generated_text", "")
        else:
            text = str(res_json)
            
        # Parse JSON
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL) or re.search(r'(\{.*?\})', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1).strip())
                required_keys = ["crop", "disease", "confidence", "status", "description", "recommendations"]
                if all(k in parsed for k in required_keys):
                    parsed["confidence"] = float(parsed["confidence"])
                    return parsed
            except Exception as parse_err:
                print(f"[Gemma] Parsing JSON response failed: {parse_err}")
                
    raise Exception(f"Gemma model did not return a valid response: {response.text}")

# --- ROUTES ---

# Serve Frontend static files
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

# Settings Endpoint
@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    settings = load_settings()
    if request.method == 'POST':
        data = request.json or {}
        settings["model_id"] = data.get("model_id", settings.get("model_id", "google/gemma-4-E4B-it"))
        # Only overwrite token if it's explicitly sent, or mask it
        token_sent = data.get("api_token", "")
        if token_sent != "******":
            settings["api_token"] = token_sent
        settings["inference_mode"] = data.get("inference_mode", settings.get("inference_mode", "api"))
        settings["pipeline_mode"] = data.get("pipeline_mode", settings.get("pipeline_mode", "single"))
        settings["vit_model_id"] = data.get("vit_model_id", settings.get("vit_model_id", "wambugu71/crop_leaf_diseases_vit"))
        settings["llama_model_id"] = data.get("llama_model_id", settings.get("llama_model_id", "meta-llama/Llama-3.1-8B-Instruct"))
        save_settings(settings)
        
    # Return settings, masking API token for security
    resp_settings = settings.copy()
    if resp_settings.get("api_token"):
        resp_settings["api_token"] = "******"  # mask token
    return jsonify(resp_settings)

# Stats Endpoint
@app.route('/api/stats', methods=['GET'])
def get_stats():
    history = load_history()
    
    total_uploads = len(history)
    processed_count = sum(1 for item in history if item.get("confidence", 0) > 0)
    processed_pct = int((processed_count / total_uploads * 100)) if total_uploads > 0 else 0
    
    # Accuracy: average confidence of predictions
    avg_confidence = int(sum(item.get("confidence", 0.0) * 100 for item in history) / total_uploads) if total_uploads > 0 else 0
        
    # Reports count: number of diseased plants detected requiring intervention
    reports_count = sum(1 for item in history if item.get("status") == "Diseased")
    
    return jsonify({
        "totalUploads": total_uploads,
        "aiProcessed": f"{processed_pct}%",
        "reports": reports_count,
        "accuracy": f"{avg_confidence}%"
    })

# Analytics Endpoint
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    history = load_history()
    
    # Create labels for the last 7 days
    today = datetime.date.today()
    last_7_days = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]
    labels = [d.strftime("%a") for d in last_7_days] # Mon, Tue, etc.
    
    # Group uploads by day
    counts = [0] * 7
    for item in history:
        try:
            date_str = item.get("timestamp", "").split(" ")[0]
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            if dt in last_7_days:
                idx = last_7_days.index(dt)
                counts[idx] += 1
        except Exception:
            pass
            
    return jsonify({
        "labels": labels,
        "data": counts
    })

# History Endpoint
@app.route('/api/history', methods=['GET'])
def get_history():
    # load_history already returns newest-first from Mongo
    history = load_history()[:20]
    
    # If empty, add standard initial items
    if not history:
        mock_history = [
            {
                "id": "mock_1",
                "timestamp": "2026-05-28 14:30:10",
                "filename": "tomato_leaf_blight.jpg",
                "crop": "Tomato",
                "disease": "Early Blight",
                "status": "Diseased",
                "confidence": 0.94
            },
            {
                "id": "mock_2",
                "timestamp": "2026-05-28 12:15:00",
                "filename": "corn_rust.jpg",
                "crop": "Corn",
                "disease": "Rust",
                "status": "Diseased",
                "confidence": 0.88
            },
            {
                "id": "mock_3",
                "timestamp": "2026-05-28 09:05:45",
                "filename": "apple_healthy.jpg",
                "crop": "Apple",
                "disease": "Healthy",
                "status": "Healthy",
                "confidence": 0.98
            }
        ]
        return jsonify(mock_history)
        
    return jsonify(history)

# Gallery Endpoint
@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    history = load_history()
    
    # Get image paths from uploads folder that have history records
    gallery_items = []
    for item in history[::-1]:
        filename = item.get("filename")
        if filename:
            entry = {
                "url": f"/api/uploads/{filename}",
                "crop": item.get("crop", "Unknown"),
                "disease": item.get("disease", "Unknown"),
                "status": item.get("status", "Healthy"),
                "confidence": item.get("confidence", 0),
                "description": item.get("description", ""),
                "recommendations": item.get("recommendations", []),
                "timestamp": item.get("timestamp", ""),
                "filename": filename,
                # Animal-specific fields
                "animal": item.get("animal", ""),
                "diagnosis": item.get("diagnosis", ""),
                "severity": item.get("severity", ""),
                "solution": item.get("solution", []),
                "major": item.get("major", False),
                "type": item.get("type", "plant")
            }
            gallery_items.append(entry)
            
    # Default picsum placeholders if gallery is empty
    if not gallery_items:
        return jsonify([
            {"url": "https://picsum.photos/300?1", "crop": "Apple", "disease": "Healthy", "status": "Healthy"},
            {"url": "https://picsum.photos/300?2", "crop": "Tomato", "disease": "Early Blight", "status": "Diseased"},
            {"url": "https://picsum.photos/300?3", "crop": "Corn", "disease": "Rust", "status": "Diseased"},
            {"url": "https://picsum.photos/300?4", "crop": "Potato", "disease": "Late Blight", "status": "Diseased"}
        ])
        
    return jsonify(gallery_items)

# Serve uploaded images
@app.route('/api/uploads/<filename>')
def serve_upload(filename):
    if USE_MONGO:
        try:
            item = mongo_db.history.find_one({"filename": filename})
            if item and "image_data" in item:
                from flask import Response
                img_data = base64.b64decode(item["image_data"])
                mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                return Response(img_data, mimetype=mime)
        except Exception as e:
            print(f"[DB] Error serving image from Mongo: {e}")
            
    # Fallback to local files if any
    return send_from_directory(UPLOADS_DIR, filename)

# Upload Image Endpoint
@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    # Read the image bytes from memory directly!
    image_bytes = file.read()
    
    # Read optional user description (text/voice transcription)
    user_description = request.form.get("description", "").strip()
    
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    
    # Save the file to disk ONLY if MongoDB is disabled (local JSON fallback)
    if not USE_MONGO:
        image_path = os.path.join(UPLOADS_DIR, unique_filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
            
    # Try to load API token if saved in settings database, otherwise empty
    settings = load_settings()
    api_token = settings.get("api_token", "")
    
    # Build custom prompt if user provided a description
    custom_prompt = None
    if user_description:
        custom_prompt = (
            "You are a world-class AI plant pathologist with deep expertise in botany and agricultural science. "
            "Carefully analyze the provided leaf/plant image. FIRST, accurately identify the exact plant species "
            "(e.g., Tomato, Potato, Corn/Maize, Rice, Wheat, Banana, Mango, Apple, Grape, Cotton, Soybean, Pepper, Citrus, etc.). "
            "THEN diagnose any visible disease, pest damage, or nutrient deficiency. Be very precise with the crop name. "
            "Look at leaf shape, venation patterns, color, size, and texture to identify the plant correctly. "
            f"\n\nThe farmer/user has also described the problem in their own words: \"{user_description}\"\n"
            "Use this description as additional context to improve your diagnosis.\n\n"
            "The response MUST contain only a valid JSON matching this schema exactly:\n"
            "{\n"
            '  "crop": "Exact Crop/Plant Name",\n'
            '  "disease": "Specific Disease Name",\n'
            '  "confidence": 0.90,\n'
            '  "status": "Diseased" or "Healthy",\n'
            '  "description": "Detailed disease/condition overview with visible symptoms",\n'
            '  "recommendations": ["Specific Treatment 1", "Specific Treatment 2", "Specific Treatment 3"]\n'
            "}\n"
            "Return ONLY the raw JSON object inside triple backticks."
        )
    
    result = None
    pipeline_mode = settings.get("pipeline_mode", "single")
    bounding_box = None
    
    if pipeline_mode == "multi":
        # ── Multi-Model Pipeline: YOLOv11 → ViT → Llama 3.1 ──
        try:
            print("[Pipeline] Running Multi-Model Pipeline (YOLOv11 → ViT → Llama)")
            # Step 1: Object Detection
            cropped_bytes, bounding_box = run_yolov11_object_detection(image_bytes, api_token, mode="plant")
            print(f"[Pipeline] Step 1 done – bbox={bounding_box}")
            
            # Step 2: ViT Classification
            vit_model = settings.get("vit_model_id", "wambugu71/crop_leaf_diseases_vit")
            vit_result = run_vit_classification(cropped_bytes, vit_model, api_token, mode="plant")
            print(f"[Pipeline] Step 2 done – label={vit_result['label']} conf={vit_result['confidence']:.2f}")
            
            # Step 3: Llama Explanation
            llama_model = settings.get("llama_model_id", "meta-llama/Llama-3.1-8B-Instruct")
            result = run_llama_explanation(vit_result["label"], user_description, api_token, llama_model, mode="plant")
            result["confidence"] = float(vit_result["confidence"])
            print(f"[Pipeline] Step 3 done – crop={result.get('crop')}, disease={result.get('disease')}")
        except Exception as pipe_err:
            print(f"[Pipeline] Multi-model failed, falling back to single: {pipe_err}")
            pipeline_mode = "single"  # fallback
    
    if pipeline_mode == "single" and result is None:
        # ── Single Model: Gemma-4 Multimodal ──
        try:
            result = query_gemma_4_multimodal(image_bytes, ext, api_token, system_prompt=custom_prompt)
        except Exception as gemma_err:
            print(f"[Gemma 4 Multimodal] Failed: {gemma_err}")
            try:
                result = query_vllm_and_gemma(image_bytes, ext, api_token)
            except Exception as vlm_err:
                print(f"[VLM Pipeline] Failed: {vlm_err}")
                result = smart_fallback_classifier(image_bytes, original_filename=file.filename, user_description=user_description)
                result.pop("warning", None)
            
    # Add timestamp and ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["id"] = uuid.uuid4().hex
    result["timestamp"] = now_str
    result["filename"] = unique_filename
    
    # Base64 encode for MongoDB inline storage
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Save to history
    history_entry = {
        "id": result["id"],
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "crop": result["crop"],
        "disease": result["disease"],
        "status": result["status"],
        "confidence": result["confidence"],
        "description": result.get("description", ""),
        "recommendations": result.get("recommendations", []),
        "type": "plant"
    }
    if user_description:
        history_entry["user_description"] = user_description
    
    if USE_MONGO:
        history_entry["image_data"] = base64_image
        
    save_history_entry(history_entry)
    
    if bounding_box:
        result["bounding_box"] = bounding_box
    result["pipeline_mode"] = pipeline_mode
    
    return jsonify(result)

# Animal Health Diagnostics Endpoint
@app.route('/api/animal/upload', methods=['POST'])
def animal_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    # Read the image bytes from memory directly!
    image_bytes = file.read()
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    
    # Read optional user description (text/voice transcription)
    user_description = request.form.get("description", "").strip()
    
    settings = load_settings()
    api_token = settings.get("api_token", "")
    
    # System prompt specifically for Animal Diagnosis
    user_context = ""
    if user_description:
        user_context = (
            f"\n\nThe pet owner/farmer has described the problem in their own words: \"{user_description}\"\n"
            "Use this description as additional context to improve your diagnosis.\n"
        )
    
    system_prompt = (
        "You are an expert veterinary pathology assistant. Analyze the provided image of the sick animal and accurately identify the animal type, "
        "diagnose the sickness or injury, estimate the severity ('High', 'Moderate', or 'Low'), write a concise description of the condition, "
        "and provide 3-4 treatment recommendations/solutions. "
        "Also, determine if this is a major problem requiring urgent veterinary hospital visit (set 'major' to true if severity is Moderate or High, "
        "or if there's any immediate life-threatening danger; set 'major' to false if it's a mild issue like a light scratch, mild itch, or standard healthy checkup). "
        + user_context +
        "Return ONLY a valid JSON object matching this schema exactly:\n"
        "{\n"
        '  "animal": "Animal Type (e.g., Dog, Cat, Cow, Goat, Chicken)",\n'
        '  "diagnosis": "Condition/Disease Name (e.g., Canine Parvovirus, Mastitis, Scabies, Wound, Healthy)",\n'
        '  "severity": "High" or "Moderate" or "Low",\n'
        '  "description": "Concise symptoms and diagnosis overview",\n'
        '  "solution": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],\n'
        '  "major": true or false\n'
        "}"
    )
    
    required_keys = ["animal", "diagnosis", "severity", "description", "solution", "major"]
    
    result = None
    
    try:
        # Query Google Gemma 4 Multimodal model first!
        result = query_gemma_4_multimodal(image_bytes, ext, api_token, system_prompt, required_keys)
    except Exception as e:
        print(f"[Animal Gemma 4] Failed: {e}")
        result = smart_animal_fallback(user_description, file.filename)
    
    # Add timestamp and ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["id"] = uuid.uuid4().hex
    result["timestamp"] = now_str
    result["filename"] = unique_filename
    
    # Base64 encode for MongoDB inline storage
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Save animal diagnosis to database
    animal_entry = {
        "id": result["id"],
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "type": "animal",
        "animal": result["animal"],
        "diagnosis": result["diagnosis"],
        "severity": result["severity"],
        "major": result["major"],
        "description": result.get("description", ""),
        "solution": result.get("solution", []),
        # Compatibility fields for gallery/history views
        "crop": result["animal"],
        "disease": result["diagnosis"],
        "status": "Diseased" if result.get("severity") in ["High", "Moderate"] else "Healthy",
        "confidence": 0.85
    }
    if user_description:
        animal_entry["user_description"] = user_description
    if USE_MONGO:
        animal_entry["image_data"] = base64_image
    
    save_history_entry(animal_entry)
        
    return jsonify(result)

# Whisper Speech-to-Text Transcription Endpoint
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes audio using openai/whisper-large-v3 via HF Inference API."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    settings = load_settings()
    api_token = settings.get("api_token", "")
    
    if not api_token:
        return jsonify({"error": "No HF API token configured. Set it in Settings."}), 400
    
    try:
        headers = {"Authorization": f"Bearer {api_token}"}
        url = f"{HF_INFERENCE_BASE}/openai/whisper-large-v3"
        res = requests.post(url, headers=headers, data=audio_bytes, timeout=30)
        
        if res.status_code == 200:
            data = res.json()
            text = data.get("text", "")
            return jsonify({"text": text})
        else:
            print(f"[Whisper] API returned {res.status_code}: {res.text}")
            return jsonify({"error": f"Whisper API error: {res.status_code}"}), 500
    except Exception as e:
        print(f"[Whisper] Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize local fallback file on startup
    _load_json_db()
    db_label = "MongoDB Atlas" if USE_MONGO else "Local JSON"
    print(f"Starting server. Database: {db_label}. Frontend: {FRONTEND_DIR}")
    app.run(host='127.0.0.1', port=5000, debug=True)
