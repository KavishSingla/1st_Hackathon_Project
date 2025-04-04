from flask import Flask, render_template, redirect, request, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from functools import wraps
from transformers import pipeline
import fitz
import re



basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SECRET_KEY"] = "Your secret key"
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


db = SQLAlchemy(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager()

login_manager.init_app(app)
login_manager.login_view = "login"
summarizer = pipeline("summarization")

class User(db.Model, UserMixin):

    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password_hash = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    

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
    

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))    
    



with app.app_context():
    db.create_all()


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def summarize_text(text):
    max_chunk = 1024
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = " ".join([summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] for chunk in text_chunks])
    return summary


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_pdf():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            text = extract_text_from_pdf(file_path)
            summary = summarize_text(text)
            pdf_file = PDFFile(filename=file.filename, text_content=text, summary=summary, user_id=current_user.id)
            db.session.add(pdf_file)
            db.session.commit()
            flash("File uploaded and summarized successfully!", "success")
            return redirect(url_for("view_pdf", pdf_id=pdf_file.id))
    return render_template("upload.html")

@app.route("/pdf/<int:pdf_id>")
@login_required
def view_pdf(pdf_id):
    pdf_file = db.session.get(PDFFile, pdf_id)
    if not pdf_file or pdf_file.user_id != current_user.id:
        flash("Unauthorized access!", "danger")
        return redirect(url_for("home"))
    return render_template("pdf_view.html", pdf=pdf_file)


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html",user = current_user)



@app.route("/contactinfo" , methods=["POST"])
def contact_info():
    flash("Thanks for your feedback", "success")
    return redirect(url_for("home"))


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
