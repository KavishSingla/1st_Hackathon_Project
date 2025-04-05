from flask import Flask, render_template, redirect, request, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from functools import wraps
from transformers import pipeline
import pdfplumber
from docx import Document
import spacy
import json
from werkzeug.utils import secure_filename
import fitz
import re
import PyPDF2
from PyPDF2 import PdfReader
from fraud_detection import detect_fraudulent_clauses


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SECRET_KEY"] = "Your secret key"
# app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



db = SQLAlchemy(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager()

login_manager.init_app(app)
login_manager.login_view = "login"

nlp = spacy.load("en_core_web_sm")


try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Error loading summarizer model: {e}")
    summarizer = None


class User(db.Model, UserMixin):

    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password_hash = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    # pdfs = db.relationship('PDFFile', backref='user', lazy=True)
    

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    
class PDFFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
class DocumentFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    entities = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))    
    



with app.app_context():
    db.create_all()


#------------ FRAUD-----------

def extract_clauses_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    clauses = [line.strip() for line in text.split('\n') if line.strip()]
    return clauses


@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    if 'pdf' not in request.files:
        return 'No PDF uploaded.'

    file = request.files['pdf']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    clauses = extract_clauses_from_pdf(path)
    results = detect_fraudulent_clauses(clauses)

    return render_template('fraud_results.html', results=results)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file with improved extraction quality.
    """
    text = ""
    try:
        # Try with PyPDF2 first
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Process each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n\n"  # Add double newline between pages
        
        # If we got very little text, there might be an issue with the extraction
        if len(text.strip()) < 100:
            # Log the issue but continue with what we have
            print("Warning: Extracted text is very short. PDF might be scanned or have restricted permissions.")
            
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace excess newlines
        text = re.sub(r'\s{2,}', ' ', text)     # Replace excess spaces
            
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = f"Error extracting text: {str(e)}"
    
    return text



def summarize_text(text):
    """
    Summarize the given text using the loaded model with enhanced detail and length.
    """
    if not text or len(text) < 100:
        return "Text too short to summarize."
    
    if not summarizer:
        return "Summarization model not available."
    
    try:
        # For longer texts, use an improved chunking approach
        max_chunk_length = 1024  # Max input size the model can handle
        min_summary_length = 300  # Increased minimum length for more detailed summary
        max_summary_length = 2000  # Significantly increased maximum length
        
        # Clean the text of extra whitespace and newlines
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is very long, process it in chunks with more detail
        if len(cleaned_text) > max_chunk_length:
            # Split text into chunks that the model can process
            chunks = [cleaned_text[i:i+max_chunk_length] 
                     for i in range(0, len(cleaned_text), max_chunk_length)]
            
            # Summarize each chunk with more detail
            chunk_summaries = []
            for i, chunk in enumerate(chunks[:5]):  # Increased from 3 to 5 chunks for more content
                chunk_summary = summarizer(
                    chunk,
                    max_length=max_summary_length//3,  # Proportional summary length
                    min_length=min_summary_length//3,
                    do_sample=False,
                    num_beams=4,  # Using beam search for better quality
                    length_penalty=2.0,  # Encourage longer summaries
                )
                # Add section header for better organization in longer summaries
                section_header = f"Part {i+1}: " if i > 0 else ""
                chunk_summaries.append(section_header + chunk_summary[0]['summary_text'])
            
            # Combine the summaries with better transitions
            combined_summary = " ".join(chunk_summaries)
            
            # For very long texts, avoid the second summarization to preserve details
            if len(chunks) <= 2:
                # Generate a final summary of the combined summaries for coherence
                final_summary = summarizer(
                    combined_summary,
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                )
                return final_summary[0]['summary_text']
            else:
                # For very long texts, just return the combined summaries to preserve detail
                return combined_summary
        else:
            # For shorter texts, summarize directly with more detail
            summary = summarizer(
                cleaned_text,
                max_length=max_summary_length,
                min_length=min_summary_length,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
            )
            return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return f"Error generating summary: {str(e)}"





@app.route('/upload_pdf', methods=['GET', 'POST'])
@login_required
def upload_pdf():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(url_for('home'))
        
        file = request.files['file']
        
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(url_for('home'))
        
        if file and allowed_file(file.filename):
            # Secure the filename to prevent security issues
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Extract text and generate summary
            extracted_text = extract_text_from_pdf(filepath)
            summary = summarize_text(extracted_text)
            
            # Save to database if user is logged in
            if current_user.is_authenticated:
                pdf_file = PDFFile(
                    filename=filename,
                    text_content=extracted_text,
                    summary=summary,
                    user_id=current_user.id
                )
                db.session.add(pdf_file)
                db.session.commit()
                flash("File uploaded and summarized successfully!", "success")
                return redirect(url_for("view_pdf", pdf_id=pdf_file.id))
            else:
                # For non-authenticated users, just show the summary without saving
                return render_template('pdf_view.html', pdf={
                    'filename': filename,
                    'text_content': extracted_text,
                    'summary': summary
                })
        else:
            flash("Only PDF files are allowed", "danger")
            return redirect(url_for('home'))
    
    # GET request - redirect to home page with upload form
    return redirect(url_for('home'))




@app.route("/pdf/<int:pdf_id>")
@login_required
def view_pdf(pdf_id):
    pdf_file = db.session.get(PDFFile, pdf_id)
    if not pdf_file or pdf_file.user_id != current_user.id:
        flash("Unauthorized access!", "danger")
        return redirect(url_for("home"))
    return render_template("pdf_view.html", pdf=pdf_file)


@app.route("/my_pdfs")
@login_required
def my_pdfs():
    pdfs = PDFFile.query.filter_by(user_id=current_user.id).all()
    return render_template("my_pdfs.html", pdfs=pdfs)


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return None
    return text.strip()

def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("home"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("home"))
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    text_content = extract_text(file_path)
    if not text_content:
        flash("Failed to extract text", "danger")
        return redirect(url_for("home"))
    
    entities = extract_entities(text_content)
    
    new_doc = DocumentFile(
        filename=filename,
        file_path=file_path,
        text_content=text_content,
        entities=str(entities),
        user_id=current_user.id
    )
    db.session.add(new_doc)
    db.session.commit()
    
    flash("File uploaded and processed successfully!", "success")
    return redirect(url_for("view_document", doc_id=new_doc.id))

@app.route("/document/<int:doc_id>")
@login_required
def view_document(doc_id):
    document = DocumentFile.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        flash("Unauthorized access!", "danger")
        return redirect(url_for("home"))

    try:
        parsed_entities = json.loads(document.entities.replace("'", '"'))
    except Exception:
        parsed_entities = {}

    return render_template("document_view.html", doc=document, entities=parsed_entities)

@app.route("/history")
@login_required
def document_history():
    documents = DocumentFile.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", documents=documents)




@app.route("/")
def home():
    return render_template("index.html")


@app.route('/translator')
# @login_required
def translator():
    return render_template('translate.html')


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html",user = current_user)



@app.route("/contact")
def contact():
    # flash("Thanks for your feedback", "success")
    return render_template('contact.html')

@app.route("/about")
def about():
    # flash("Thanks for your feedback", "success")
    return render_template('about.html')

@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')


@app.route("/apply")
@login_required
def apply():
    flash("Thanks for applying the card , details will be shared on your email ", "info")
    return redirect(url_for("home"))



@app.errorhandler(404)
def page_not_found(e):
  print(e) 
  return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        mobile = request.form.get("mobile")
        gender = request.form.get("gender")
        

      
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

       
        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for("register"))
        
        if not re.match(r"^(?=.*[!@#$%^&*(),.?\":{}|<>])(?=.*\d)[A-Za-z\d!@#$%^&*(),.?\":{}|<>]{8,}$", password):
            flash("Password must be at least 8 characters long, include one number and one special character.", "danger")
            return redirect(url_for("register"))
        
        new_user = User(name=name, email=email, mobile=mobile, gender = gender)
        new_user.set_password(password) 
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")




@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        

        user = User.query.filter_by(email=email , name = name).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials!", "danger")

    return render_template("login.html")
           




@app.route("/update/<int:id>", methods=["GET", "POST"])
@login_required
def update(id):
    user = db.session.get(User, id)
    if request.method == "POST":
        name = request.form.get("name")
        gender = request.form.get("gender")
        mobile = request.form.get("mobile")
        

        user.name = name
        user.gender = gender
        user.mobile = mobile
        

        db.session.add(user)
        db.session.commit()
        flash("Profile Updated", "info")
        return redirect(url_for("profile"))

    return render_template("update_user.html", user = user)
    


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "info")
    return redirect(url_for("home"))


@app.route("/delete/<int:id>")
@login_required

def delete(id):
    user = db.session.get(User, id)
    db.session.delete(user)
    db.session.commit()
    flash("Profile deleted successfully!", "success")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
