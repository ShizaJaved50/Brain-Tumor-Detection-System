# app.py
from flask import Flask, request, render_template,  redirect, flash, session,url_for
# from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
load_dotenv()
import time
from model import CNNModel
import torch
from torchvision import transforms
from PIL import Image
import os
from model import db, User, DiagnosisResult
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from flask import make_response
import matplotlib.pyplot as plt
import time
# from model import predict
app = Flask(__name__)
from flask_migrate import Migrate
timestamp = int(time.time())
migrate = Migrate(app,db)
app.secret_key = os.getenv("SECRET_KEY")
# PostgreSQL DB connection
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'postgresql://postgres:root@localhost:5432/Flask_login_db'



db.init_app(app)

with app.app_context():
    db.create_all()
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        flash("Username and password are required", "danger")
        return redirect(url_for('index'))  # index page

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        session['username'] = user.username
        flash("Login successful!", "success")
        return redirect(url_for('Form')) 
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for('index'))
# Optional: Register new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash("Username and password are required", "danger")
            return redirect('/')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect('/')

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registered successfully!', 'success')
        return redirect('/Form')

    # Optional: render a registration page if it's a GET request
    return render_template('register.html')



model = CNNModel()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/results')
def results():
    return render_template("results.html")

# Preprocess the image function
def preprocess_image(image):
    if image.mode == 'L':  # Convert grayscale to RGB
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 224x224 (same as in your Jupyter notebook)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Prediction function
def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output)  # Apply sigmoid to get probabilities
        prediction = (probability > 0.5).float()  # Apply threshold
        return "Tumor Detected" if prediction.item() == 1 else "No Tumor Detected"

# Flask route
@app.route('/Form', methods=['GET', 'POST'])
def Form():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file uploaded", "danger")
            return redirect('/')

        file = request.files['image']
        if file:
            img = Image.open(file).convert('RGB')  # Convert image to RGB if necessary
            image_tensor = preprocess_image(img)  # Preprocess image
            result = predict_image(model, image_tensor)  # Predict result

            # Get additional form data
            form_data = {
                "predicted_class": result,
                "confidence": f"{torch.sigmoid(model(image_tensor)).max().item():.2%}",
                "pain": request.form.get("pain"),
                "history": request.form.get("history"),
                "pain_level": request.form.get("pain_level"),
                "pain_duration": request.form.get("pain_duration"),
                "symptoms": request.form.get("symptoms"),
                "family_history": request.form.get("family_history"),
            }

        confidence = float(form_data["confidence"].strip('%')) / 100
        labels = ['Tumor', 'No Tumor']
        values = [confidence, 1 - confidence]
        colors = ['#ff4d4d', '#66cc66']


        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values, color=colors)
        plt.ylim(0, 1)  # Set y-axis from 0 to 1 (0% to 100%)
        plt.title("Tumor Prediction Confidence")
        plt.xlabel("Category")
        plt.ylabel("Confidence")
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value*100:.1f}%', ha='center', va='bottom')
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        graph_image = base64.b64encode(buffer.read()).decode('utf-8')
        # Save to database
        if 'username' in session:
             save_result_to_db(
                username=session['username'],
                predicted_class=form_data["predicted_class"],
                confidence=confidence,
                # graph_image_base64=graph_image,
                symptoms=form_data["symptoms"],
                pain_level=form_data["pain_level"],
                family_history=form_data["family_history"]
            )

        
        return render_template("results.html", result=form_data, graph_image=graph_image, timestamp=timestamp)
    return render_template('Form.html')

def save_result_to_db(username, predicted_class, confidence, symptoms, pain_level, family_history):
    result = DiagnosisResult(
        username=username,
        predicted_class=predicted_class,
        confidence=confidence,
        # graph_image=graph_image_base64,
        symptoms=symptoms,
        pain_level=pain_level,
        family_history=family_history
    )
    db.session.add(result)
    db.session.commit()
    

@app.route('/download_pdf')
def download_pdf():
    if 'username' not in session:
        flash("Please login first to download the report", "warning")
        return redirect(url_for('index'))

    result = DiagnosisResult.query.filter_by(username=session['username']).order_by(DiagnosisResult.id.desc()).first()

    if not result:
        flash("No diagnosis record found.", "info")
        return redirect(url_for('Form'))

    pdf_buffer = BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Medical Diagnosis Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 450, f"Diagnosis: {result.predicted_class}")
    p.drawString(100, 430, f"Confidence: {result.confidence}")
    p.drawString(100, 410, f"Pain Level: {result.pain_level}")
    p.drawString(100, 390, f"Symptoms: {result.symptoms}")
    p.drawString(100, 370, f"Family History: {result.family_history}")

    p.save()
    pdf_buffer.seek(0)

    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=diagnosis_report.pdf'
    return response
if __name__ == "__main__":
    app.run(debug=True)
    with app.app_context():
        db.create_all()
    