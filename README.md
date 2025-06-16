# 🔍 Sistema de Reconhecimento Facial - Scikit-learn

Um sistema **leve e eficiente** de reconhecimento facial que identifica se uma pessoa em uma foto é você ou outra pessoa, usando **Machine Learning clássico** com Scikit-learn.

## 🌟 Por que Scikit-learn ao invés de TensorFlow?

- ⚡ **Mais rápido:** Treinamento em segundos, não minutos
- 💾 **Mais leve:** Modelo de ~5MB vs ~50MB
- 🔧 **Mais simples:** Sem dependências pesadas do TensorFlow
- 🎯 **Eficiente:** 85-95% de precisão com muito menos recursos
- 💻 **CPU-friendly:** Funciona bem sem GPU

## 🚀 Características

- **🧠 Machine Learning Clássico:** Random Forest, SVM, Gradient Boosting
- **🔍 Extração de Características:** HOG, LBP, Estatísticas e Pixels
- **📸 Data Augmentation:** Rotação, flip, brilho, contraste
- **🌐 Interface Web:** Frontend moderno e responsivo
- **⚡ API REST:** Backend Flask otimizado
- **📊 Múltiplos Algoritmos:** Testa 5 algoritmos e escolhe o melhor
- **🎯 Alta Precisão:** 85-95% de accuracy típica
- **💻 Cross-platform:** Windows, Mac, Linux

## 🏗️ Arquitetura

### Extração de Características
```
Imagem (128x128) → Pré-processamento → Extração de Features
                                           ↓
                                    [HOG Features]
                                    [LBP Histogram]  
                                    [Pixel Features]
                                    [Stats Features]
                                           ↓
                                    Vector Combinado
                                           ↓
                                      Classificador
                                           ↓
                                    Resultado + Confiança
```

### Algoritmos Testados
- **Random Forest** (Recomendado)
- **SVM** com kernel RBF
- **Gradient Boosting**
- **K-Nearest Neighbors**
- **Logistic Regression**

O sistema testa todos e escolhe automaticamente o melhor!

## 💾 Instalação

### 1. Clonar Repositório
```bash
git clone https://github.com/DenisSMorais2/Sistema-de-Reconhecimento-Facial---Scikit-learn.git
cd face-recognition-sklearn
```

### 2. Criar Ambiente Virtual (RECOMENDADO)
```bash
# Windows
python -m venv sklearn_env
sklearn_env\Scripts\activate

# Linux/Mac
python3 -m venv sklearn_env
source sklearn_env/bin/activate
```

### 3. Instalar Dependências
```bash
# Atualizar pip primeiro
python -m pip install --upgrade pip

# Instalar todas as dependências
pip install -r requirements.txt

# OU instalar uma por uma se der erro:
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install flask flask-cors
pip install pillow numpy joblib
pip install matplotlib seaborn
```

### 4. Verificar Instalação
```bash
python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"
python -c "import cv2; print('✅ OpenCV:', cv2.__version__)"
python -c "from skimage import feature; print('✅ Scikit-image: OK')"
```

### 🐛 Se Houver Erro de Dependências

**Erro: "No module named 'cv2'"**
```bash
pip install opencv-python
```

**Erro: "No module named 'skimage'"**
```bash
pip install scikit-image
# OU se der erro de compilação:
conda install scikit-image
```

**Erro: Visual C++ Build Tools**
- Baixe e instale: **Microsoft C++ Build Tools**
- OU use conda: `conda install scikit-image`

## 📊 Quick Start (3 minutos)

### Resultado Real de Exemplo
Com o dataset de **246 imagens** (120 suas + 126 outras), o sistema alcançou:

```
📊 Dataset processado:
   - Total de imagens: 246
   - Classe 'aluno': 120
   - Classe 'outros': 126  
   - Características por imagem: 9,155

🏆 Resultado do Treinamento:
   - Melhor modelo: Random Forest
   - Accuracy no teste: 100.0%
   - Precision: 100% (Aluno e Outros)
   - Recall: 100% (Aluno e Outros)
   - F1-Score: 100% (Aluno e Outros)

📊 Matriz de Confusão:
   - Verdadeiros Positivos: 24/24 (100%)
   - Verdadeiros Negativos: 26/26 (100%)
   - Falsos Positivos: 0
   - Falsos Negativos: 0
```

### Método Mais Rápido
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Coletar fotos (50+ suas + 100+ outras)
python photo_collector.py

# 3. Processar dataset
python create_dataset.py

# 4. Treinar modelo (30 segundos!)
python train_model.py

# 5. Executar sistema
python app.py
```

**⚡ Tempo total: ~5 minutos para setup + coleta de fotos**

Acesse: **http://localhost:5000** 🚀

## 📁 Estrutura do Projeto

```
face-recognition-sklearn/
│
├── 📁 dataset/                    # Dataset original
│   ├── aluno/                     # Suas fotos (50+)
│   └── outros/                    # Outras pessoas (100+)
│
├── 📁 processed_data/             # Dados processados
│   ├── features.npy               # Características extraídas
│   └── labels.npy                 # Labels das classes
│
├── 📁 templates/                  # Frontend
│   └── index.html                 # Interface web moderna
│
├── 🐍 photo_collector.py          # Coletar fotos com webcam
├── 🐍 create_dataset.py           # Processar dataset + features
├── 🐍 train_model.py              # Treinar modelo ML
├── 🐍 app.py                      # Backend Flask
│
├── 🤖 face_recognition_sklearn_model.pkl  # Modelo treinado
├── 📊 model_info.pkl              # Info do modelo
├── 📄 requirements.txt            # Dependências leves
└── 📋 README.md                   # Este arquivo
```

## 🎯 Uso Detalhado

### 1. Coletar Fotos
```bash
python photo_collector.py
```

**Menu interativo:**
- Opção 1: Capturar suas fotos (50 recomendado)
- Opção 2: Capturar fotos de outras pessoas (100+ recomendado)
- Opção 3: Verificar quantidade atual
- Opção 4: Prosseguir para treinamento

**Dicas para captura:**
- 📸 Varie poses: frontal, perfil, 3/4
- 😊 Diferentes expressões: sério, sorrindo
- 💡 Iluminações variadas: natural, artificial
- 👓 Com/sem acessórios: óculos, boné, barba

### 2. Processar Dataset
```bash
python create_dataset.py
```

**O que faz:**
- Aplica data augmentation (rotação, flip, brilho)
- Extrai características HOG + LBP + Pixels + Stats
- Salva features processadas em NumPy arrays
- Otimizado para velocidade

### 3. Treinar Modelo
```bash
python train_model.py
```

**Processo automático:**
- Testa 5 algoritmos diferentes (Random Forest, SVM, Gradient Boosting, KNN, Logistic Regression)
- Cross-validation com 5 folds para cada modelo
- Otimização automática de hiperparâmetros
- Escolhe o melhor modelo baseado na accuracy
- Salva modelo otimizado (.pkl)

**Saída típica real:**
```
🤖 TREINAMENTO DE MODELO - SCIKIT-LEARN
📊 Dados carregados: 246 amostras, 9155 características

🔄 Avaliando modelos...
📊 Testando Random Forest...     Accuracy: 1.000 (+/- 0.000)
📊 Testando SVM...              Accuracy: 1.000 (+/- 0.000)  
📊 Testando Gradient Boosting... Accuracy: 1.000 (+/- 0.000)
📊 Testando KNN...              Accuracy: 1.000 (+/- 0.000)
📊 Testando Logistic Regression... Accuracy: 1.000 (+/- 0.000)

🏆 Melhor modelo: Random Forest
🔧 Otimizando hiperparâmetros...
   Melhores parâmetros: {'rf__max_depth': 10, 'rf__min_samples_leaf': 1, 
                         'rf__min_samples_split': 2, 'rf__n_estimators': 50}

🎯 RESULTADO FINAL:
   Accuracy no teste: 100.0%
   Precision (Outros): 100%    Recall (Outros): 100%
   Precision (Aluno): 100%     Recall (Aluno): 100%

📊 MATRIZ DE CONFUSÃO:
   Verdadeiros Negativos: 26    Falsos Positivos: 0
   Falsos Negativos: 0          Verdadeiros Positivos: 24

💾 Modelo salvo como: face_recognition_sklearn_model.pkl
```

**⚡ Tempo de treinamento: ~30-60 segundos**

### 4. Executar Sistema
```bash
python app.py
```

**Endpoints disponíveis:**
- `GET /` - Interface web
- `POST /predict` - Predição de imagem
- `GET /health` - Status da API
- `GET /model-info` - Informações do modelo
- `GET /test-model` - Teste automático

## 🔌 API Documentation

### Predição de Imagem
```http
POST /predict
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

### Resposta
```json
{
    "is_student": true,
    "prediction": 1,
    "confidence": 0.923,
    "probabilities": {
        "others": 0.077,
        "student": 0.923
    },
    "message": "É você! (Confiança: 92.3%)",
    "model_info": {
        "type": "Random Forest",
        "accuracy": 0.915
    }
}
```

### Health Check
```http
GET /health
```

```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "Random Forest",
    "accuracy": 0.915
}
```

## 🔍 Características Extraídas

### 1. HOG (Histogram of Oriented Gradients)
- **O que é:** Descreve formas e bordas
- **Parâmetros:** 9 orientações, células 8x8
- **Dimensões:** ~3000+ features

### 2. LBP (Local Binary Patterns)
- **O que é:** Texturas locais da face
- **Parâmetros:** 24 pontos, raio 8
- **Dimensões:** 26 bins

### 3. Pixel Features
- **O que é:** Intensidades dos pixels
- **Dimensões:** 32x32 = 1024 features
- **Normalização:** 0-1

### 4. Estatísticas
- **O que é:** Média, desvio, mediana, min, max
- **Dimensões:** 5 features

**Total:** ~4000+ características por imagem

## ⚙️ Personalização

### Alterar Algoritmo Principal
```python
# Em train_model.py, modifique create_models()
models = {
    'SVM Otimizado': Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma=0.01, probability=True))
    ])
}
```

### Modificar Extração de Features
```python
# Em create_dataset.py, função extract_features()
hog_features = feature.hog(
    gray, 
    orientations=12,      # Mais orientações
    pixels_per_cell=(4, 4), # Células menores
    cells_per_block=(3, 3)  # Blocos maiores
)
```

### Ajustar Data Augmentation
```python
# Em create_dataset.py, função data_augmentation()
transformations = [
    cv2.flip(img, 1),     # Flip horizontal
    rotate_image(img, 30), # Mais rotação
    adjust_brightness(img, 1.5), # Mais brilho
    add_noise(img)        # Adicionar ruído
]
```

## 📊 Benchmarks

### Performance Real Alcançada
Com dataset de **246 imagens** (120 suas + 126 outras pessoas):

```
📈 RESULTADOS REAIS DO SISTEMA:
Algoritmo           | Accuracy | Cross-Val | Otimização
--------------------|----------|-----------  |------------
Random Forest ⭐    | 100.0%   | 1.000±0.000| ✅ Melhor
SVM (RBF)           | 100.0%   | 1.000±0.000| ✅ Excelente  
Gradient Boosting   | 100.0%   | 1.000±0.000| ✅ Excelente
KNN                 | 100.0%   | 1.000±0.000| ✅ Excelente
Logistic Regression | 100.0%   | 1.000±0.000| ✅ Excelente

🏆 MODELO FINAL (Random Forest):
- Accuracy: 100.0%
- Precision: 100% (ambas as classes)
- Recall: 100% (ambas as classes)  
- F1-Score: 100% (ambas as classes)
- Zero falsos positivos/negativos
- Características: 9,155 por imagem
- Tempo de treinamento: ~45 segundos
- Tempo de predição: ~50ms por imagem
```

### Performance Típica vs Real
```
Cenário               | Esperado | Alcançado | Observação
----------------------|----------|-----------|------------
Dataset Pequeno       | 85-90%   | 100.0%    | ⭐ Excelente
Dataset Médio         | 90-95%   | 100.0%    | ⭐ Perfeito
Tempo Treino          | 30-60s   | ~45s      | ✅ Conforme
Tempo Predição        | ~50ms    | ~50ms     | ✅ Rápido
Features Extraídas    | ~4000    | 9,155     | ⭐ Mais rico
Balanceamento Classes | Bom      | 120/126   | ✅ Ideal
```

### Teste de Confiança Real
```
🧪 AMOSTRAS DE TESTE:
Amostra 1: Real=Aluno ➜ Predito=Aluno    (Confiança: 98.0%)
Amostra 2: Real=Aluno ➜ Predito=Aluno    (Confiança: 100.0%)  
Amostra 3: Real=Outros ➜ Predito=Outros  (Confiança: 94.0%)
Amostra 4: Real=Aluno ➜ Predito=Aluno    (Confiança: 90.0%)
Amostra 5: Real=Outros ➜ Predito=Outros  (Confiança: 100.0%)
```

### Comparação com Deep Learning
```
Método              | Accuracy | Modelo Size | Tempo Treino | Dataset | CPU/GPU
--------------------|----------|-------------|--------------|---------|--------
Este Sistema ⭐     | 100.0%   | ~5MB       | 45s          | 246 img | CPU
CNN (TensorFlow)    | 92-98%   | ~50MB       | 10-30min     | 1000+   | GPU
Face Recognition Lib| 95-99%   | ~100MB      | N/A          | N/A     | CPU/GPU

💡 OBSERVAÇÃO: Nosso sistema alcançou 100% de accuracy com apenas 246 imagens,
   superando expectativas! Isso demonstra a eficácia das características 
   extraídas (9,155 features) e da qualidade do dataset balanceado.
```

## 🐛 Troubleshooting

### Problemas Comuns

#### ❌ "No module named 'sklearn'"
```bash
pip install scikit-learn==1.3.2
```

#### ❌ "OpenCV not found"
```bash
pip install opencv-python==4.8.1.78
```

#### ❌ "Low accuracy (< 70%)"
**Soluções:**
- ✅ Adicione mais fotos (100+ suas, 200+ outros)
- ✅ Use fotos com melhor qualidade
- ✅ Varie mais as condições de captura
- ✅ Verifique se as fotos têm faces bem visíveis

#### ❌ "Model file not found"
```bash
# Certifique-se de treinar primeiro
python train_model.py
```

#### ❌ "Webcam not working"
```python
# Teste diferentes índices
cap = cv2.VideoCapture(1)  # Tente 0, 1, 2...
```

#### ❌ "Out of memory"
```python
# Reduza o tamanho das imagens em create_dataset.py
img_resized = cv2.resize(img_rgb, (64, 64))  # Menor que 128x128
```

### Logs de Debug
```python
# Adicionar em app.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Para ver detalhes do treinamento
print(f"Features shape: {X.shape}")
print(f"Labels distribution: {np.bincount(y)}")
```

## 🎉 Caso de Sucesso Real

### 📊 Dataset Usado
- **Total de imagens:** 246
- **Suas fotos:** 120 (classe "aluno")  
- **Outras pessoas:** 126 (classe "outros")
- **Características extraídas:** 9,155 por imagem
- **Balanceamento:** Quase perfeito (120/126)

### 🏆 Resultados Alcançados
- **Accuracy:** 100.0% (perfeito!)
- **Modelo escolhido:** Random Forest
- **Tempo de treinamento:** ~45 segundos
- **Zero erros:** Nem falsos positivos nem negativos
- **Confiança média:** 94-100%

### 💡 Fatores do Sucesso
1. **Dataset bem balanceado** (120 vs 126 imagens)
2. **Qualidade das fotos** coletadas  
3. **Características ricas** (9,155 features por imagem)
4. **Data augmentation** aplicado corretamente
5. **Otimização automática** de hiperparâmetros

Este resultado demonstra que o **Scikit-learn pode superar Deep Learning** em cenários específicos com datasets de qualidade! 🚀

### 1. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Selecionar melhores features
selector = SelectKBest(f_classif, k=1000)
X_selected = selector.fit_transform(X, y)
```

### 2. Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

# Combinar múltiplos modelos
ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True)),
    ('gb', GradientBoostingClassifier())
], voting='soft')
```

### 3. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Reduzir dimensionalidade
pca = PCA(n_components=500)
X_reduced = pca.fit_transform(X)
```

## 🎯 Melhorias Futuras

### V2.0 Planejado
- [ ] 🎥 Reconhecimento via webcam em tempo real
- [ ] 📱 Versão mobile com kivy/BeeWare
- [ ] 🔐 Sistema multi-usuário
- [ ] 📊 Dashboard com métricas detalhadas
- [ ] 🌐 Deploy em cloud (Heroku/Railway)
- [ ] 🎭 Detecção de múltiplas faces
- [ ] 🚀 Otimização com ONNX
- [ ] 📈 Active learning para melhorar modelo

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-feature`
3. Commit: `git commit -m 'Adiciona nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra um Pull Request

## 📈 Comparação Detalhada

### Scikit-learn vs TensorFlow

| Aspecto          | Scikit-learn  | TensorFlow      |
|---------         |-------------- |------------     |
| **Instalação**   | 50MB          | 500MB+          |
| **Tempo Treino** | 30 segundos   | 10-30 minutos   |
| **Accuracy**     | 90-95%        | 92-98%          |
| **Modelo Size**  | 5MB           | 50MB            |
| **RAM Usage**    | 500MB         | 2GB+            |
| **CPU vs GPU**   | CPU eficiente | GPU recomendada |
| **Complexidade** | Simples       | Complexa        |
| **Manutenção**   | Fácil         | Difícil         |

### Quando Usar Cada Um?

**Use Scikit-learn quando:**
- ✅ Prototipagem rápida
- ✅ Recursos limitados (CPU/RAM)
- ✅ Dataset pequeno/médio (< 10k imagens)
- ✅ Simplicidade é prioridade
- ✅ Accuracy 90%+ é suficiente

**Use TensorFlow quando:**
- ✅ Accuracy máxima é crucial
- ✅ Dataset muito grande (100k+ imagens)
- ✅ GPU disponível
- ✅ Recursos computacionais abundantes
- ✅ Aplicação crítica

## Resultados
### Minha foto:

![Captura de tela 2025-06-16 121452](https://github.com/user-attachments/assets/92a5369f-a9b0-4652-abd1-aaf6a5248aea)

### Outra pessoa sem direitos autorais:

![no_face](https://github.com/user-attachments/assets/af99821c-d052-4d89-a43d-66baeec5c523)
