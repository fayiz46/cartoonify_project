# Cartoonify Image Web App 🚀

A FastAPI-based web application to upload and cartoonify images using OpenCV.  
The app features a simple, user-friendly HTML interface with image upload and beautiful results display.

---

## ✨ Features
- Upload an image and get a cartoonified version.
- FastAPI backend with OpenCV image processing.
- User-friendly frontend with sky-blue background and light green buttons.
- Dockerized for easy deployment.

---

## 🛠️ Technologies Used
- Python 3.9
- FastAPI
- Jinja2 (for HTML templating)
- OpenCV (opencv-python-headless)
- Docker

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cartoonify-app.git
cd cartoonify-app
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
uvicorn main:app --reload
Open your browser and visit:
http://localhost:8000

🐳 Running with Docker
1. Build Docker Image
bash
Copy
Edit
docker build -t cartoonify-app .
2. Run Docker Container
bash
Copy
Edit
docker run -d -p 8000:8000 --name cartoonify-container cartoonify-app
3. Access the App
text
Copy
Edit
http://localhost:8000
📂 Project Structure
arduino
Copy
Edit
cartoonify-app/
│
├── Dockerfile
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── uploads/ (image storage)
└── .dockerignore
✅ Requirements
All Python dependencies are listed in requirements.txt.

💬 License
This project is open source and free to use.