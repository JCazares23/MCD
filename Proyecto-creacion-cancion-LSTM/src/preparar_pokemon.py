import glob
import os
from music21 import converter, instrument, note, chord

# Localización automática de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_RAW = os.path.join(BASE_DIR, "data", "raw_pokemon_music")
RUTA_PROCESSED = os.path.join(BASE_DIR, "data", "processed_pokemon_music")

def extraer_notas_carpeta(ruta_busqueda):
    notas_totales = []
    archivos = glob.glob(os.path.join(ruta_busqueda, "*.mid"))
    print(f"Encontrados {len(archivos)} archivos MIDI en {ruta_busqueda}")

    for archivo in archivos:
        try:
            midi = converter.parse(archivo)
            partes = instrument.partitionByInstrument(midi)
            notas_a_leer = partes.parts[0].recurse() if partes else midi.flat.notes
            for elemento in notas_a_leer:
                if isinstance(elemento, note.Note):
                    notas_totales.append(str(elemento.pitch))
                elif isinstance(elemento, chord.Chord):
                    notas_totales.append('.'.join(str(n) for n in elemento.normalOrder))
        except Exception as e:
            print(f"Error en {os.path.basename(archivo)}: {e}")
            continue
    return " ".join(notas_totales)

if __name__ == "__main__":
    if not os.path.exists(RUTA_PROCESSED):
        os.makedirs(RUTA_PROCESSED)
    
    datos = extraer_notas_carpeta(RUTA_RAW)
    with open(os.path.join(RUTA_PROCESSED, "pokemon_final_data.txt"), "w") as f:
        f.write(datos)
    print(f"Éxito: Datos guardados en data/processed_pokemon_music/")