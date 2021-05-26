from flask import Flask, render_template, flash, request, redirect, url_for
from markupsafe import escape
import os
from werkzeug.utils import secure_filename
from birdsong_recognition.inference import inference

app = Flask(__name__)

# Set limit on upload size to 16M
app.config['MAX_CONTENT_LENGTH'] = 16E6

UPLOAD_FOLDER = './test_data/norcar/'
ALLOWED_EXTENSIONS = {'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/uploads/<name>")
def uploaded_file(name):
    predicted_bird, confidence_level = inference(name)

    return render_template('inference.html', name=name, prediction=predicted_bird, confidence=confidence_level)