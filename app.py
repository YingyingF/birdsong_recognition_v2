from flask import Flask, render_template, flash, request, redirect, url_for, Response
from markupsafe import escape
import os
import pathlib
from werkzeug.utils import secure_filename
import tempfile
from birdsong_recognition.inference import inference

app = Flask(__name__)

# Set limit on upload size to 16M
app.config['MAX_CONTENT_LENGTH'] = 16E6

UPLOAD_FOLDER = './test_data/norcar/'
pathlib.Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
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
    <h1>Upload bird song </h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/play/<name>")
def playback(name):
    data = open(pathlib.Path(UPLOAD_FOLDER) / name, 'rb').read()
    return Response(data, mimetype='audio/mp3')

@app.route("/uploads/<name>")
def uploaded_file(name):
    predicted_bird, confidence_level = inference(name)

    return render_template('inference.html', name=name, prediction=predicted_bird, confidence=confidence_level)

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')