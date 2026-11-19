from flask import Flask
from dotenv import load_dotenv
import os


def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'svsm-dev-secret')
    app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')
    app.config['RESULTS_FOLDER'] = os.path.abspath('results')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    from .routes import bp
    app.register_blueprint(bp)
    return app
