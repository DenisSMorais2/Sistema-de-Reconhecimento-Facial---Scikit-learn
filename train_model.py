import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Carrega os dados processados"""
    try:
        X = np.load('processed_data/features.npy')
        y = np.load('processed_data/labels.npy')
        print(f"📊 Dados carregados: {X.shape[0]} amostras, {X.shape[1]} características")
        return X, y
    except FileNotFoundError:
        print("❌ Dados não encontrados! Execute 'python create_dataset.py' primeiro")
        return None, None

def create_models():
    """Cria diferentes modelos para comparação"""
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]),
        
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    return models

def evaluate_models(X, y):
    """Avalia diferentes modelos usando cross-validation"""
    models = create_models()
    results = {}
    
    print("🔄 Avaliando modelos...")
    print("="*60)
    
    for name, model in models.items():
        print(f"📊 Testando {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"   Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Encontrar melhor modelo
    best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Melhor modelo: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['cv_mean']:.3f}")
    
    return best_model, best_model_name, results

def hyperparameter_tuning(model, X, y):
    """Otimiza hiperparâmetros do melhor modelo"""
    print("\n🔧 Otimizando hiperparâmetros...")
    
    # Parâmetros para Random Forest (modelo mais comum)
    if 'RandomForestClassifier' in str(model):
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [10, 20, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }
    elif 'SVC' in str(model):
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    else:
        print("   Usando parâmetros padrão")
        return model
    
    # Grid search
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print(f"   Melhor score: {grid_search.best_score_:.3f}")
    print(f"   Melhores parâmetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def train_final_model(X, y):
    """Treina o modelo final"""
    print("\n🎯 Treinando modelo final...")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dados de treino: {X_train.shape[0]} amostras")
    print(f"📊 Dados de teste: {X_test.shape[0]} amostras")
    
    # Avaliar modelos
    best_model, best_name, all_results = evaluate_models(X_train, y_train)
    
    # Otimizar hiperparâmetros
    optimized_model = hyperparameter_tuning(best_model, X_train, y_train)
    
    # Treinar modelo final
    optimized_model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = optimized_model.predict(X_test)
    y_pred_proba = optimized_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   Modelo: {best_name}")
    print(f"   Accuracy no teste: {accuracy:.3f}")
    print(f"   Total de amostras: {len(X)}")
    
    # Relatório detalhado
    print(f"\n📋 RELATÓRIO DETALHADO:")
    print(classification_report(y_test, y_pred, target_names=['Outros', 'Aluno']))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 MATRIZ DE CONFUSÃO:")
    print(f"   Verdadeiros Negativos (Outros): {cm[0][0]}")
    print(f"   Falsos Positivos (Outros como Aluno): {cm[0][1]}")
    print(f"   Falsos Negativos (Aluno como Outros): {cm[1][0]}")
    print(f"   Verdadeiros Positivos (Aluno): {cm[1][1]}")
    
    # Salvar modelo
    model_filename = 'face_recognition_sklearn_model.pkl'
    joblib.dump(optimized_model, model_filename)
    print(f"\n💾 Modelo salvo como: {model_filename}")
    
    # Salvar informações adicionais
    model_info = {
        'model_name': best_name,
        'accuracy': accuracy,
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    joblib.dump(model_info, 'model_info.pkl')
    print("💾 Informações do modelo salvas como: model_info.pkl")
    
    return optimized_model, accuracy

def plot_results(results):
    """Plota resultados da comparação de modelos"""
    try:
        models = list(results.keys())
        accuracies = [results[model]['cv_mean'] for model in models]
        stds = [results[model]['cv_std'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, yerr=stds, capsize=5, 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'khaki', 'plum'])
        
        plt.title('Comparação de Modelos - Accuracy com Cross-Validation')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico salvo como 'model_comparison.png'")
        
    except Exception as e:
        print(f"⚠️ Erro ao criar gráfico: {e}")

def main():
    """Função principal"""
    print("🤖 TREINAMENTO DE MODELO - SCIKIT-LEARN")
    print("="*50)
    
    # Carregar dados
    X, y = load_data()
    if X is None:
        return
    
    # Verificar balanceamento
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Distribuição das classes:")
    print(f"   Classe 0 (Outros): {counts[0]} amostras")
    print(f"   Classe 1 (Aluno): {counts[1]} amostras")
    
    if len(unique) < 2:
        print("❌ Erro: Necessário pelo menos 2 classes! Adicione fotos de outras pessoas.")
        return
    
    # Treinar modelo
    model, accuracy = train_final_model(X, y)
    
    print(f"\n✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Accuracy final: {accuracy:.1%}")
    print(f"   Modelo salvo e pronto para uso!")
    
    # Testar predição
    print(f"\n🧪 TESTE RÁPIDO:")
    sample_X = X[:5]  # Primeiras 5 amostras
    sample_y = y[:5]
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)
    
    for i in range(len(sample_X)):
        real_class = "Aluno" if sample_y[i] == 1 else "Outros"
        pred_class = "Aluno" if predictions[i] == 1 else "Outros"
        confidence = probabilities[i].max()
        print(f"   Amostra {i+1}: Real={real_class}, Predito={pred_class}, Confiança={confidence:.1%}")

if __name__ == "__main__":
    main()