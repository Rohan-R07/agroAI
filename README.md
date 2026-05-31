# 🌱 AgroAI - Intelligent Plant & Animal Health Monitoring Platform

AgroAI is a modern, responsive, and beautiful AI-powered diagnostic platform designed for modern agriculture. It provides end-to-end multi-model pipelines for real-time plant disease detection, animal health diagnostics, nearby veterinary care integration, voice explanations, and detailed agricultural report generation.

---

## ✨ Features

- **🌱 Plant Health Diagnostic Pipeline**: 
  - **Step 1 (YOLOv11)**: Automaticaly localizes and crops leaf areas for analysis.
  - **Step 2 (ViT Classifier)**: Utilizes a high-precision `MobileNet V2` ViT model to classify crop species and disease.
  - **Step 3 (Llama 3.1 Reasoning)**: Generates structured, readable diagnosis and actionable recommendations.
- **🐾 Animal Health Diagnostic Pipeline**: 
  - Employs a BioCLIP model to diagnose symptoms for common farm and household animals.
- **🗺️ Nearby Veterinarian Care Finder**:
  - Leverages Leaflet & CartoDB Maps to automatically locate and display veterinary clinics based on the user's location.
- **🌓 Premium Dark & Light Mode Theme**:
  - Features a highly polished, responsive toggle theme that persists across pages and device reloads.
- **📈 PDF Report Generator**:
  - Automatically compiles AI diagnostics, treatment steps, and clinic details into a professional PDF report.
- **🎙️ Voice Explanation (TTS)**:
  - Explains the agricultural recommendations to farmers in multiple languages.

---

## 🛠️ Tech Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+), Leaflet Maps, jsPDF.
- **Backend**: Flask (Python), PyMongo, MongoDB Atlas, Pillow, OpenAI API Client.
- **AI Models**: Hugging Face Serverless Router Infrastructure (`router.huggingface.co`).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- MongoDB Atlas account (or runs on automatic local JSON database fallback)

### 2. Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rohan-R07/agroAI.git
   cd agroAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_api_token_here
   ```

4. **Run the Application**:
   ```bash
   python backend/app.py
   ```

---

## 🔒 Security
- All sensitive credentials and local user upload data are explicitly ignored in `.gitignore`.
- API keys are loaded dynamically via environment configurations to prevent public exposure.
