import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import pickle
from skimage import feature
import joblib

def create_folders():
    """Cria as pastas necessárias"""
    folders = ['dataset/aluno', 'dataset/outros', 'processed_data']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Pastas criadas!")

def extract_features(image_path):
    """Extrai características da imagem usando HOG e LBP"""
    # Carregar imagem
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Converter para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar para tamanho padrão
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # 1. HOG Features (Histogram of Oriented Gradients)
    hog_features = feature.hog(
        gray, 
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    # 2. LBP Features (Local Binary Patterns)
    lbp = feature.local_binary_pattern(gray, P=24, R=8, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalizar
    
    # 3. Pixel intensities (redimensionado)
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
    
    # Combinar todas as características
    combined_features = np.concatenate([
        hog_features,
        lbp_hist,
        pixel_features,
        stats_features
    ])
    
    return combined_features

def process_dataset():
    """Processa o dataset e extrai características"""
    print("🔄 Processando dataset...")
    
    features = []
    labels = []
    
    # Processar fotos do aluno (classe 1)
    aluno_path = 'dataset/aluno'
    if os.path.exists(aluno_path):
        for filename in os.listdir(aluno_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(aluno_path, filename)
                img_features = extract_features(img_path)
                if img_features is not None:
                    features.append(img_features)
                    labels.append(1)  # Classe "aluno"
                    print(f"✅ Processado: {filename}")
    
    # Processar fotos de outros (classe 0)
    outros_path = 'dataset/outros'
    if os.path.exists(outros_path):
        for filename in os.listdir(outros_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(outros_path, filename)
                img_features = extract_features(img_path)
                if img_features is not None:
                    features.append(img_features)
                    labels.append(0)  # Classe "outros"
                    print(f"✅ Processado: {filename}")
    
    if not features:
        print("❌ Nenhuma imagem encontrada! Adicione fotos nas pastas dataset/aluno e dataset/outros")
        return None, None
    
    # Converter para numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Embaralhar dados
    X, y = shuffle(X, y, random_state=42)
    
    print(f"📊 Dataset processado:")
    print(f"   - Total de imagens: {len(X)}")
    print(f"   - Classe 'aluno': {np.sum(y == 1)}")
    print(f"   - Classe 'outros': {np.sum(y == 0)}")
    print(f"   - Características por imagem: {X.shape[1]}")
    
    # Salvar dados processados
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/features.npy', X)
    np.save('processed_data/labels.npy', y)
    
    print("💾 Dados salvos em 'processed_data/'")
    
    return X, y

def data_augmentation():
    """Aplica data augmentation básico"""
    print("🔄 Aplicando data augmentation...")
    
    for category in ['aluno', 'outros']:
        input_dir = f'dataset/{category}'
        
        if not os.path.exists(input_dir):
            continue
            
        for img_file in os.listdir(input_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(input_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Aplicar transformações
                transformations = [
                    # Flip horizontal
                    cv2.flip(img, 1),
                    # Rotação +15°
                    cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1), (img.shape[1], img.shape[0])),
                    # Rotação -15°
                    cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -15, 1), (img.shape[1], img.shape[0])),
                    # Ajuste de brilho
                    cv2.convertScaleAbs(img, alpha=1.2, beta=20),
                    # Ajuste de contraste
                    cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
                ]
                
                # Salvar versões aumentadas
                base_name = os.path.splitext(img_file)[0]
                for i, transformed in enumerate(transformations):
                    aug_filename = f'{base_name}_aug_{i+1}.jpg'
                    aug_path = os.path.join(input_dir, aug_filename)
                    cv2.imwrite(aug_path, transformed)
                
                print(f"✅ Augmentation aplicado em: {img_file}")
    
    print("🎯 Data augmentation concluído!")

if __name__ == "__main__":
    print("🔥 PROCESSADOR DE DATASET - SCIKIT-LEARN")
    print("="*50)
    
    create_folders()
    
    print("\n1. Coloque suas fotos na pasta 'dataset/aluno'")
    print("2. Coloque fotos de outras pessoas na pasta 'dataset/outros'")
    print("3. Execute este script para processar o dataset")
    
    choice = input("\nAplicar data augmentation? (s/n): ").lower().strip()
    if choice == 's':
        data_augmentation()
    
    X, y = process_dataset()
    
    if X is not None:
        print("\n✅ Processamento concluído!")
        print("📋 Próximo passo: Execute 'python train_model.py' para treinar o modelo")