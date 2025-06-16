import cv2
import os
import time
from datetime import datetime

def create_folders():
    """Cria as pastas necessárias"""
    folders = ['dataset/aluno', 'dataset/outros']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Pastas criadas!")

def capture_photos(category="aluno", num_photos=50):
    """Captura fotos usando a webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível acessar a webcam")
        return
    
    print(f"📸 Capturando {num_photos} fotos para: {category}")
    print("\n🎯 INSTRUÇÕES:")
    print("- Pressione ESPAÇO para tirar foto")
    print("- Pressione ESC para sair")
    print("- Varie as poses e expressões!")
    
    if category == "aluno":
        print("\n💡 DICAS PARA SUAS FOTOS:")
        print("• Frontal, perfil direito, perfil esquerdo")
        print("• Sorrindo, sério, surpreso")
        print("• Com/sem óculos, diferentes roupas")
        print("• Diferentes iluminações")
    
    count = 0
    folder_path = f'dataset/{category}'
    
    print(f"\n🚀 Começando em 3 segundos...")
    time.sleep(3)
    
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Espelhar a imagem
        frame = cv2.flip(frame, 1)
        
        # Interface na tela
        cv2.putText(frame, f'Fotos: {count}/{num_photos}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Categoria: {category.upper()}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, 'ESPACO=Foto  ESC=Sair', (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progresso visual
        progress_width = int((count / num_photos) * frame.shape[1])
        cv2.rectangle(frame, (0, 0), (progress_width, 5), (0, 255, 0), -1)
        
        cv2.imshow('📸 Captura de Fotos', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espaço para tirar foto
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f'{folder_path}/{category}_{count+1:03d}_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            count += 1
            print(f"📸 Foto {count}/{num_photos} salva!")
            
            # Feedback visual (flash)
            flash_frame = frame.copy()
            cv2.rectangle(flash_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 20)
            cv2.imshow('📸 Captura de Fotos', flash_frame)
            cv2.waitKey(150)
            
        elif key == 27:  # ESC para sair
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Captura concluída! {count} fotos salvas em '{folder_path}'")

def main_menu():
    """Menu principal simplificado"""
    create_folders()
    
    while True:
        print("\n" + "="*50)
        print("📸 COLETOR DE FOTOS - SCIKIT-LEARN")
        print("="*50)
        print("1. 🎯 Capturar SUAS fotos (aluno)")
        print("2. 👥 Capturar fotos de OUTRAS pessoas")
        print("3. 📊 Verificar quantas fotos tenho")
        print("4. 🚀 Prosseguir para o treinamento")
        print("5. ❌ Sair")
        
        choice = input("\nEscolha uma opção (1-5): ").strip()
        
        if choice == '1':
            num_photos = input("Quantas fotos suas capturar? (recomendado: 50): ").strip()
            num_photos = int(num_photos) if num_photos.isdigit() else 50
            print(f"\n🎯 Preparando para capturar {num_photos} SUAS fotos...")
            capture_photos("aluno", num_photos)
            
        elif choice == '2':
            num_photos = input("Quantas fotos de outras pessoas? (recomendado: 100): ").strip()
            num_photos = int(num_photos) if num_photos.isdigit() else 100
            print(f"\n👥 Preparando para capturar {num_photos} fotos de outras pessoas...")
            print("💡 Peça para amigos/família posarem para você!")
            capture_photos("outros", num_photos)
            
        elif choice == '3':
            aluno_count = len([f for f in os.listdir('dataset/aluno') if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('dataset/aluno') else 0
            outros_count = len([f for f in os.listdir('dataset/outros') if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('dataset/outros') else 0
            
            print(f"\n📊 RESUMO DO DATASET:")
            print(f"   🎯 Suas fotos: {aluno_count}")
            print(f"   👥 Outras pessoas: {outros_count}")
            print(f"   📈 Total: {aluno_count + outros_count}")
            
            if aluno_count >= 30 and outros_count >= 50:
                print("   ✅ Dataset está bom para treinamento!")
            elif aluno_count < 30:
                print(f"   ⚠️ Adicione mais {30 - aluno_count} fotos suas")
            elif outros_count < 50:
                print(f"   ⚠️ Adicione mais {50 - outros_count} fotos de outras pessoas")
                
        elif choice == '4':
            aluno_count = len([f for f in os.listdir('dataset/aluno') if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('dataset/aluno') else 0
            outros_count = len([f for f in os.listdir('dataset/outros') if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('dataset/outros') else 0
            
            if aluno_count >= 10 and outros_count >= 10:
                print("\n🚀 Dataset pronto! Execute os comandos:")
                print("   1. python create_dataset.py")
                print("   2. python train_model.py")
                print("   3. python app.py")
                break
            else:
                print("\n❌ Dataset insuficiente!")
                print(f"   Você tem: {aluno_count} suas fotos, {outros_count} de outros")
                print("   Mínimo: 10 suas fotos, 10 de outras pessoas")
                
        elif choice == '5':
            print("👋 Até logo!")
            break
            
        else:
            print("❌ Opção inválida!")

if __name__ == "__main__":
    main_menu()