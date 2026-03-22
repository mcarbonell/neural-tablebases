import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

def build_shards(input_dir, output_dir, samples_per_shard=4_000_000, num_shards=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar solo los datasets canonical
    npz_files = glob.glob(os.path.join(input_dir, "*_canonical.npz"))
    if not npz_files:
        print(f"No *_canonical.npz files found in {input_dir}")
        return
        
    print(f"Encontrados {len(npz_files)} datasets. Construyendo {num_shards} shards de ~{samples_per_shard} posiciones...")
    
    # 1. Cargar metadatos mmap de los datasets para ver su tamaño y la dimensión de 'x'
    datasets = []
    print("Abriendo datasets en modo mmap (bajo consumo RAM)...")
    for f in tqdm(npz_files):
        try:
            d = np.load(f, mmap_mode='r')
            datasets.append({
                'name': os.path.basename(f),
                'x': d['x'],
                'wdl': d['wdl'],
                'dtz': d['dtz'],
                'size': len(d['wdl']),
                'dim': d['x'].shape[1]
            })
        except Exception as e:
            print(f"Error cargando {f}: {e}")
            
    if not datasets:
        return
        
    num_datasets = len(datasets)
    samples_per_dataset = samples_per_shard // num_datasets
    
    # 2. Descubrir la dimensión máxima (Max Padding)
    max_dim = max(d['dim'] for d in datasets)
    print(f"\n--- APLICANDO PADDING: Dimensión final universal = {max_dim} variables ---")
    
    # 3. Creación de Shards (Generadores ciegos con Oversampling de dataset pequeños)
    dataset_indices = {i: np.random.permutation(d['size']) for i, d in enumerate(datasets)}
    dataset_cursors = {i: 0 for i in range(num_datasets)}
    
    for shard_idx in range(num_shards):
        print(f"\n[+] Construyendo Shard {shard_idx+1}/{num_shards} ...")
        shard_x = []
        shard_wdl = []
        shard_dtz = []
        
        for i, d in enumerate(datasets):
            needed = samples_per_dataset
            cursor = dataset_cursors[i]
            indices = dataset_indices[i]
            
            idx_list = []
            while needed > 0:
                available = d['size'] - cursor
                take = min(needed, available)
                idx_list.append(indices[cursor:cursor+take])
                needed -= take
                cursor += take
                
                # Reshuffle & Wrap-around si consumimos el dataset entero (ideal para que las 3-piezas no se acaben)
                if cursor >= d['size']:
                    indices = np.random.permutation(d['size'])
                    dataset_indices[i] = indices
                    cursor = 0
                    
            dataset_cursors[i] = cursor
            
            # Extraer vectores
            extract_idx = np.concatenate(idx_list)
            
            # --- PADDING DINÁMICO ---
            x_data = d['x'][extract_idx]
            if x_data.shape[1] < max_dim:
                padding = np.zeros((x_data.shape[0], max_dim - x_data.shape[1]), dtype=x_data.dtype)
                x_data = np.concatenate([x_data, padding], axis=1)
                
            shard_x.append(x_data)
            shard_wdl.append(d['wdl'][extract_idx])
            shard_dtz.append(d['dtz'][extract_idx])
            
        print(" -> Mezclando posiciones internas...")
        full_x = np.concatenate(shard_x, axis=0)
        full_wdl = np.concatenate(shard_wdl, axis=0)
        full_dtz = np.concatenate(shard_dtz, axis=0)
        
        # Super Shuffle para el Dataloader (crucial para matar el Olvido Catastrófico)
        shuffle_idx = np.random.permutation(len(full_wdl))
        full_x = full_x[shuffle_idx]
        full_wdl = full_wdl[shuffle_idx]
        full_dtz = full_dtz[shuffle_idx]
        
        # Guardar en disco el Shard final
        out_file = os.path.join(output_dir, f"v5_universe_shard_{shard_idx+1:03d}.npz")
        # Usamos savez_compressed para no reventar el disco
        np.savez_compressed(out_file, x=full_x, wdl=full_wdl, dtz=full_dtz)
        print(f" [+] Shard Guardado: {out_file} (Total: {len(full_wdl)} posiciones).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/v5", help="Directorio con los npz_canonical")
    parser.add_argument("--output_dir", type=str, default="data/shards", help="Directorio salida de los shards")
    parser.add_argument("--samples", type=int, default=4000000, help="Posiciones totales por shard")
    parser.add_argument("--shards", type=int, default=10, help="Número de shards a generar")
    args = parser.parse_args()
    
    build_shards(args.input_dir, args.output_dir, args.samples, args.shards)
