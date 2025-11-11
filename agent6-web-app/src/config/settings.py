import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_default_secret_key'
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ['true', '1']
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'src/static/uploads'
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'png', 'jpg', 'jpeg', 'gif'}
    API_BASE_URL = os.environ.get('API_BASE_URL') or 'http://localhost:5000/api'