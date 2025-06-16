from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import cv2
import numpy as np
from PIL import Image
import io
import base64
from skimage import feature
import os

app = Flask(__name__)
CORS(app)

# Variáveis globais
model = None
model_info = None

def load_model():
    """Carrega o modelo treinado"""
    global model, model_info
    
    try:
        model = joblib.load('face_recognition_sklearn_model.pkl')
        model_info = joblib.load('model_info.pkl')
        print("✅ Modelo carregado com sucesso!")
        print(f"   Tipo: {model_info['model_name']}")
        print(f"   Accuracy: {model_info['accuracy']:.1%}")
        return True
    except FileNotFoundError:
        print("❌ Modelo não encontrado! Execute 'python train_model.py' primeiro")
        return False

def extract_features(img_array):
    """Extrai características da imagem (mesmo processo do treinamento)"""
    # Redimensionar para tamanho padrão
    img_resized = cv2.resize(img_array, (128, 128))
    
    # Converter para escala de cinza
    if len(img_resized.shape) == 3:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_resized
    
    # 1. HOG Features
    hog_features = feature.hog(
        gray, 
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # 2. LBP Features
    lbp = feature.local_binary_pattern(gray, P=24, R=8, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    # 3. Pixel intensities
    pixel_features = cv2.resize(gray, (32, 32)).flatten()
    pixel_features = pixel_features.astype(float) / 255.0
    
    # 4. Estatísticas básicas
    stats_features = np.array([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.max(gray),
        np.min(gray)
    ])
    
    # Combinar características
    combined_features = np.concatenate([
        hog_features,
        lbp_hist,
        pixel_features,
        stats_features
    ])
    
    return combined_features

def preprocess_image(image_data):
    """Processa a imagem recebida"""
    try:
        # Converter de base64 para imagem
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Converter para numpy array
        img_array = np.array(image)
        
        # Extrair características
        features = extract_features(img_array)
        
        return features.reshape(1, -1)  # Reshape para predição
        
    except Exception as e:
        print(f"Erro no preprocessamento: {e}")
        return None

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Endpoint para verificar saúde da API"""
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': model_info['model_name'] if model_info else 'Unknown',
            'accuracy': model_info['accuracy'] if model_info else 'Unknown'
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Modelo não carregado'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para fazer predições"""
    if model is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        # Receber dados
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Imagem não fornecida'}), 400
        
        # Preprocessar imagem
        features = preprocess_image(data['image'])
        if features is None:
            return jsonify({'error': 'Erro ao processar imagem'}), 400
        
        # Fazer predição
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Interpretar resultado
        if prediction == 1:
            # É o aluno
            confidence = probabilities[1]
            result = {
                'is_student': True,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'others': float(probabilities[0]),
                    'student': float(probabilities[1])
                },
                'message': f'É você! (Confiança: {confidence*100:.1f}%)',
                'model_info': {
                    'type': model_info['model_name'] if model_info else 'Unknown',
                    'accuracy': model_info['accuracy'] if model_info else 'Unknown'
                }
            }
        else:
            # Não é o aluno
            confidence = probabilities[0]
            result = {
                'is_student': False,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'others': float(probabilities[0]),
                    'student': float(probabilities[1])
                },
                'message': f'Não é você (Confiança: {confidence*100:.1f}%)',
                'model_info': {
                    'type': model_info['model_name'] if model_info else 'Unknown',
                    'accuracy': model_info['accuracy'] if model_info else 'Unknown'
                }
            }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/model-info')
def get_model_info():
    """Retorna informações sobre o modelo"""
    if model_info:
        return jsonify(model_info)
    else:
        return jsonify({'error': 'Informações do modelo não disponíveis'}), 404

@app.route('/test-model')
def test_model():
    """Endpoint para testar se o modelo está funcionando"""
    if model is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    # Criar uma imagem de teste (ruído)
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extract_features(test_image).reshape(1, -1)
    
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return jsonify({
            'test_status': 'success',
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'message': 'Modelo funcionando corretamente'
        })
    except Exception as e:
        return jsonify({
            'test_status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 INICIANDO SERVIDOR FLASK COM SCIKIT-LEARN")
    print("="*50)
    
    # Carregar modelo
    if load_model():
        print("✅ Servidor pronto!")
        print("🌐 Acesse: http://localhost:5000")
        print("📊 Health check: http://localhost:5000/health")
        print("🧪 Teste do modelo: http://localhost:5000/test-model")
        print("ℹ️ Info do modelo: http://localhost:5000/model-info")
        
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("❌ Não foi possível carregar o modelo")
        print("💡 Execute 'python train_model.py' primeiro")