"""
NeuroScan AI - Brain Tumor Detection Web Application
Flask Backend API
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import json
from werkzeug.utils import secure_filename
from utilities import focal_tversky, tversky_loss, tversky

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
classification_model = None
segmentation_model = None
models_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load pre-trained classification and segmentation models"""
    global classification_model, segmentation_model, models_loaded
    
    try:
        # Load classification model (ResNet-50)
        print("Loading classification model...")
        with open('resnet-50-MRI.json', 'r') as json_file:
            json_savedModel = json_file.read()
        # Replace 'Model' with 'Functional' for compatibility with newer Keras
        json_savedModel = json_savedModel.replace('"class_name": "Model"', '"class_name": "Functional"')
        classification_model = tf.keras.models.model_from_json(json_savedModel, custom_objects={'Functional': tf.keras.Model, 'tversky': tversky, 'tversky_loss': tversky_loss, 'focal_tversky': focal_tversky})
        classification_model.load_weights('weights.hdf5')
        classification_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=["accuracy"]
        )
        print("✓ Classification model loaded successfully")
        
        # Load segmentation model (ResUNet)
        print("Loading segmentation model...")
        with open('ResUNet-MRI.json', 'r') as json_file:
            json_savedModel = json_file.read()
        # Replace 'Model' with 'Functional' for compatibility with newer Keras
        json_savedModel = json_savedModel.replace('"class_name": "Model"', '"class_name": "Functional"')
        segmentation_model = tf.keras.models.model_from_json(json_savedModel, custom_objects={'Functional': tf.keras.Model, 'tversky': tversky, 'tversky_loss': tversky_loss, 'focal_tversky': focal_tversky})
        segmentation_model.load_weights('weights_seg.hdf5')
        adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
        segmentation_model.compile(
            optimizer=adam,
            loss=focal_tversky,
            metrics=[tversky]
        )
        print("✓ Segmentation model loaded successfully")
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        models_loaded = False
        return False


def preprocess_image_classification(img):
    """Preprocess image for classification model"""
    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1]
    img = img * 1.0 / 255.0
    
    # Convert to float64
    img = np.array(img, dtype=np.float64)
    
    # Reshape for model input
    img = np.reshape(img, (1, 256, 256, 3))
    
    return img


def preprocess_image_segmentation(img):
    """Preprocess image for segmentation model"""
    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Convert to float64
    img = np.array(img, dtype=np.float64)
    
    # Standardize (mean centering and std scaling)
    img -= img.mean()
    img /= img.std()
    
    # Reshape for model input
    X = np.empty((1, 256, 256, 3))
    X[0,] = img
    
    return X


def predict_tumor(image_path):
    """
    Two-stage prediction:
    1. Classification: Does the image have a tumor?
    2. Segmentation: If yes, where is the tumor located?
    """
    
    # Read image
    img_original = cv2.imread(image_path)
    if img_original is None:
        return None
    
    # Stage 1: Classification
    img_class = preprocess_image_classification(img_original.copy())
    classification_pred = classification_model.predict(img_class, verbose=0)
    
    has_tumor = bool(np.argmax(classification_pred) == 1)
    confidence = float(classification_pred[0][np.argmax(classification_pred)])
    
    result = {
        'has_tumor': has_tumor,
        'confidence': confidence,
        'classification_scores': {
            'no_tumor': float(classification_pred[0][0]),
            'tumor': float(classification_pred[0][1])
        }
    }
    
    # Stage 2: Segmentation (only if tumor detected)
    if has_tumor:
        img_seg = preprocess_image_segmentation(img_original.copy())
        segmentation_pred = segmentation_model.predict(img_seg, verbose=0)
        
        # Convert prediction to binary mask
        mask = segmentation_pred[0].squeeze()
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Calculate tumor area
        tumor_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        # Convert mask to base64 for frontend display
        mask_img = (mask_binary * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_img, cv2.COLORMAP_HOT)
        
        # Create overlay image
        overlay = img_original.copy()
        overlay = cv2.resize(overlay, (256, 256))
        overlay[mask_binary == 1] = [0, 255, 0]  # Green overlay
        
        # Convert images to base64
        _, mask_buffer = cv2.imencode('.png', mask_colored)
        _, overlay_buffer = cv2.imencode('.png', overlay)
        
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        result['segmentation'] = {
            'mask': f"data:image/png;base64,{mask_base64}",
            'overlay': f"data:image/png;base64,{overlay_base64}",
            'tumor_area_percentage': float(tumor_percentage),
            'tumor_pixels': int(tumor_pixels)
        }
    
    return result


# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'version': '1.0.0'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please restart the server.'
        }), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, TIF, TIFF'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and convert original image to PNG for browser compatibility
        img_original = cv2.imread(filepath)
        if img_original is None:
            # Try with PIL for TIFF support
            img_pil = Image.open(filepath)
            img_original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Convert to PNG and encode as base64
        _, img_buffer = cv2.imencode('.png', img_original)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        
        # Make prediction
        prediction_result = predict_tumor(filepath)
        
        if prediction_result is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Add original image to result
        prediction_result['original_image'] = f"data:image/png;base64,{img_base64}"
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return project statistics"""
    stats = {
        'total_patients': 110,
        'total_scans': 3929,
        'model_accuracy': 97.92,
        'segmentation_score': 0.92,
        'average_inference_time': 2.3,
        'model_type': 'ResNet-50 + ResUNet'
    }
    return jsonify(stats)


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 16MB.'}), 413


if __name__ == '__main__':
    print("=" * 60)
    print("🧠 NeuroScan AI - Brain Tumor Detection System")
    print("=" * 60)
    print("\nInitializing application...")
    
    # Load models
    if load_models():
        print("\n✓ All models loaded successfully!")
        print("\n🚀 Starting Flask server...")
        print("📍 Access the application at: http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please check that model files exist.")
        print("Required files:")
        print("  - resnet-50-MRI.json")
        print("  - weights.hdf5")
        print("  - ResUNet-MRI.json")
        print("  - weights_seg.hdf5")
        print("  - utilities.py")
