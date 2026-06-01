# 🌱 AgroAI - Intelligent Plant & Animal Health Monitoring Platform

AgroAI is a highly polished, responsive, and beautiful AI-powered diagnostic platform designed for modern agriculture and animal husbandry. It provides end-to-end intelligent vision pipelines for real-time plant crop pathology, insect infestation analysis, animal health diagnostics, nearby veterinary care integration, local history tracking, and professional agricultural report generation.

🚀 **Live Production Link:** [https://agroai-kappa.vercel.app](https://agroai-kappa.vercel.app)

---

## ✨ Features & Capabilities

### 1. 🔬 Upgraded AI Vision Diagnostics
* **Advanced SOTA VLM Engine:** Powered by **`Qwen/Qwen3-VL-235B-A22B-Instruct`**—a 235-billion-parameter state-of-the-art vision-language model. This provides exceptional accuracy for identifying plant species, crop diseases, pest damage, and nutrient deficiencies.
* **Dual Processing Pipeline:**
  * **Single Model Mode:** Direct multimodal vision analysis utilizing Qwen3-VL-235B for lightning-fast, high-context agricultural reasoning.
  * **Multi-Model Mode:** Sequential execution utilizing **YOLOv11** (leaf localization & cropping) ➡️ **ViT Classifier** (species prediction) ➡️ **Llama 3.1 8B Instruct** (actionable recommendation writing).

### 2. 📸 Native Mobile Camera & Gallery Integration
* Refactored file upload systems to include custom styled side-by-side controls:
  * **📁 Choose Image:** Allows standard local photo uploads from desktop or mobile libraries.
  * **📷 Take Photo:** Hooks directly into mobile hardware to launch the **rear (environment) camera** instantly (`capture="environment"`), making crop field analysis seamless.

### 3. 🌐 End-to-End Multilingual Support (18 Global Languages)
* Integrated native translation capabilities across:
  * **Speech Recognition:** Live voice-to-text recording (Whisper/Web Speech API) with language selection.
  * **Contextual Prompt Injection:** Injects translation commands dynamically based on selected language codes (e.g. `hi-IN` for Hindi, `bn-IN` for Bengali, `es-ES` for Spanish) so that the entire diagnosis, symptoms, and treatments are generated natively in that language by the AI!
  * **Text-to-Speech (TTS):** Read Aloud synthesis for audio feedback of treatment recommendations in the selected language.

### 4. 🐾 Animal Health Diagnostic Pipeline & Map Finder
* Seamlessly analyzes medical conditions, injuries, and severity levels in livestock and household pets.
* **Veterinary Hospital Finder:** Automatically queries location APIs to discover nearby vet clinics and maps them onto a custom Leaflet map interface when moderate/severe symptoms are detected.

### 5. 🗄️ Serverless Ready Cloud Storage (Vercel & MongoDB Atlas)
* Configured serverless execution for production environments using Vercel.
* **Dynamic Writable Path Router:** Checks runtime environments and utilizes the system's write-enabled `/tmp` directory under read-only serverless constraints (resolving file-system startup crashes).
* Integrates securely with **MongoDB Atlas** database cloud storage for persistent diagnostics history and stats.

---

## 🛠️ Tech Stack

* **Frontend:** Vanilla HTML5, Vanilla CSS3 (Custom styling grid & dark/light theme), JavaScript (ES6+), Leaflet Map & CartoDB tiles.
* **Backend:** Flask (Python), PyMongo, MongoDB Atlas, Pillow, OpenAI API Client, certifi (SSL).
* **Deployment:** Vercel Serverless Functions.

---

## 🚀 Getting Started Locally

### 1. Prerequisites
* Python 3.10+
* Hugging Face Account (for API tokens)
* MongoDB Atlas Cloud Account (or runs on automatic local JSON database fallback)

### 2. Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rohan-R07/agroAI.git
   cd agroAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a local configuration folder in `backend/data/` or set environment variables in your local environment:
   * `MONGO_URI` = your_mongodb_connection_string
   * `HF_TOKEN` = your_huggingface_api_token

4. **Run the Application:**
   ```bash
   python backend/app.py
   ```
   Open your browser and navigate to `http://localhost:5000` to interact with the system locally.

---

## ☁️ Cloud Deployment (Vercel CLI)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Link the Project:**
   ```bash
   vercel link --scope anik-da-projects1 --project agroai --token YOUR_TOKEN --yes
   ```

3. **Configure Environment Settings:**
   ```bash
   vercel env add MONGO_URI production --value "YOUR_MONGO_URI" --token YOUR_TOKEN --yes
   vercel env add HF_TOKEN production --value "YOUR_HF_TOKEN" --token YOUR_TOKEN --yes
   ```

4. **Deploy to Production:**
   ```bash
   vercel --prod --token YOUR_TOKEN --yes
   ```

---

## 🔒 Security & Standards
* Dynamic environment variable resolution.
* Sensitive local diagnostic photos and access keys are ignored in `.gitignore`.
* Production Vercel routes use SSL-certified pipelines.
