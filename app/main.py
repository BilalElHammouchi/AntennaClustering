from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a route for the root URL
@app.route('/')
def index():
    # Render the 'index.html' template
    return render_template('index.html')

# Define a route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'csvFile' not in request.files:
        return "No file part"

    file = request.files['csvFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return f"File '{file.filename}' successfully uploaded and saved at {file_path}"

if __name__ == '__main__':
    # Run the application on http://127.0.0.1:5000/
    app.run(debug=True)
