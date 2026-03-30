import os
import torch
import torch.nn as nn
import numpy as np
import random
from music21 import stream, note, chord, midi, instrument

# 1. Configuración de rutas relativas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATA = os.path.join(BASE_DIR, "data", "processed_pokemon_music", "pokemon_final_data.txt")
RUTA_MODELO = os.path.join(BASE_DIR, "models", "modelo_entrenado.pth")
RUTA_SALIDA = os.path.join(BASE_DIR, "pokemon_remix.mid")

# 2. Cargar el vocabulario (DEFINIR 'text' AQUÍ)
if not os.path.exists(RUTA_DATA):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {RUTA_DATA}")

with open(RUTA_DATA, "r") as f:
    text = f.read()  # Aquí se define la variable 'text'

notas_separadas = text.split()
vocab = sorted(set(notas_separadas))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 3. Definir la Arquitectura del Modelo
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

# 4. Cargar el "Cerebro" entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModeloMusical(len(vocab), 256, 1024).to(device)

if os.path.exists(RUTA_MODELO):
    model.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
    model.eval()
else:
    raise FileNotFoundError("No se encontró el modelo entrenado en la carpeta 'models/'.")

# 5. Lógica de Generación
def generar_melodia_pokemon(largo=300):
    punto_inicio = random.randint(0, min(500, len(notas_separadas) - 6))
    inicio = notas_separadas[punto_inicio : punto_inicio + 5]
    
    input_eval = torch.tensor([[char2idx[s] for s in inicio]]).to(device)
    notas_generadas = []
    states = None

    for _ in range(largo):
        prediccion, states = model(input_eval, states)
        prediccion = prediccion[:, -1, :]
        
        # Temperatura 1.0 para mantener balance
        prediccion = torch.softmax(prediccion / 1.0, dim=-1)
        id_predicho = torch.multinomial(prediccion, num_samples=1).item()
        
        input_eval = torch.tensor([[id_predicho]]).to(device)
        notas_generadas.append(idx2char[id_predicho])

    return notas_generadas

# 6. Guardar archivo MIDI
def salvar_a_midi(lista_notas, nombre_archivo):
    salida = stream.Stream()
    salida.append(instrument.Piano())
    
    for patron in lista_notas:
        if ('.' in patron) or patron.isdigit():
            notas_en_acorde = patron.split('.')
            nuevas_notas = []
            for n_actual in notas_en_acorde:
                nueva_nota = note.Note(int(n_actual))
                nuevas_notas.append(nueva_nota)
            nuevo_acorde = chord.Chord(nuevas_notas)
            nuevo_acorde.quarterLength = 0.5
            salida.append(nuevo_acorde)
        else:
            nueva_nota = note.Note(patron)
            nueva_nota.quarterLength = 0.5
            salida.append(nueva_nota)

    salida.write('midi', fp=nombre_archivo)
    print(f"Archivo guardado exitosamente en: {nombre_archivo}")

if __name__ == "__main__":
    print("Generando composición original...")
    melodia = generar_melodia_pokemon(largo=300)
    salvar_a_midi(melodia, RUTA_SALIDA)