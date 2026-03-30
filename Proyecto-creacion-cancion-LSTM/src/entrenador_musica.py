import torch
import torch.nn as nn
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATA = os.path.join(BASE_DIR, "data", "processed_pokemon_music", "pokemon_final_data.txt")
RUTA_MODELO = os.path.join(BASE_DIR, "models", "modelo_entrenado.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Corriendo en: {device}")

# Carga de datos con rutas relativas
with open(RUTA_DATA, "r") as f:
    text = f.read()

notas_separadas = text.split()
vocab = sorted(set(notas_separadas))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in notas_separadas])

print(f"{len(vocab)} notas únicas encontradas.")

# El modelo
class ModeloMusical(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, states=None):
        x = self.embedding(x)
        x, states = self.lstm(x, states)
        x = self.fc(x)
        return x, states

# Instancias del modelo
model = ModeloMusical(len(vocab), 256, 1024).to(device)
print("Modelo construido y listo en la memoria.")

# Probar el modelo
def predecir_siguiente(texto_semilla):
    model.eval()
    # Convertir texto a números
    input_eval = [char2idx[s] for s in texto_semilla]
    input_eval = torch.tensor([input_eval]).to(device)
    
    # Pedirle a la red que piense
    prediccion, _ = model(input_eval)
    
    # Tomar la nota con mas probabilidad
    ultima_prediccion = prediccion[0, -1, :]
    indice_predicho = torch.argmax(ultima_prediccion).item()
    
    return idx2char[indice_predicho]

# Entrenamiento
# Criterio de error y optimizado con ADAM
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\nIniciando entrenamiento (100 EPOCHS)...")

# Convertimos todo el texto a numeros para que la red estudie
input_seq = torch.tensor([text_as_int[:-1]]).to(device) # Todas las notas menos la última
target_seq = torch.tensor([text_as_int[1:]]).to(device)  # Todas las notas menos la primera

model.train()
for epoch in range(100):
    optimizer.zero_grad()

    output, _ = model(input_seq)
    
    loss = criterion(output.transpose(1, 2), target_seq)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"EPOCH {epoch+1}/100 - Error: {loss.item():.4f}")

if not os.path.exists(os.path.dirname(RUTA_MODELO)):
    os.makedirs(os.path.dirname(RUTA_MODELO))
torch.save(model.state_dict(), RUTA_MODELO)
print(f"Modelo guardado en: models/modelo_entrenado.pth")