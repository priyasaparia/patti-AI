# 🌿 Patti AI — Setup Guide

## Folder Structure
```
patti-ai/
├── app.py                  ← Flask backend
├── requirements.txt
├── plant_disease_model.h5  ← Put your trained model here
├── class_names.txt         ← One class name per line (optional)
└── static/
    └── index.html          ← Frontend UI
```

## Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 2 — Install & start MongoDB
```bash
# Ubuntu/Debian
sudo apt install mongodb
sudo systemctl start mongodb

# macOS
brew install mongodb-community
brew services start mongodb-community

# Windows: download from https://www.mongodb.com/try/download/community
```

## Step 3 — Add your trained model
Copy your saved Keras model to this folder:
```bash
cp /path/to/your/plant_disease_model.h5 ./plant_disease_model.h5
```

Also create `class_names.txt` with one class per line matching your model's output:
```
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
...
```

## Step 4 — Run the server
```bash
python app.py
```

## Step 5 — Open in browser
```
http://localhost:5000
```

---

## Features
- ✅ Signup / Login / Logout
- ✅ Upload image (drag & drop or browse)
- ✅ Camera capture (OpenCV via browser)
- ✅ AI diagnosis with confidence bar
- ✅ Previous searches history (MongoDB)
- ✅ Beautiful cream + green UI
"# pattiai" 
