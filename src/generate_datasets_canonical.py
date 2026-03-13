"""
Generador de datasets con canonical forms.
Modificación de generate_datasets_parallel.py para usar canonical forms.
"""
import chess
import chess.syzygy
import numpy as np
import os
import argparse
from typing import List, Tuple
import itertools
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from datetime import timedelta
import pickle

# Importar funciones existentes
sys.path.append(os.path.dirname(__file__))
from generate_datasets import encode_board, piece_move_distance, encode_board_relative

# Importar canonical forms
try:
    from canonical_forms import find_canonical_form, board_to_encoding_key
except ImportError:
    # Si no está disponible, crear funciones dummy
    print("WARNING: canonical_forms module not found. Using dummy functions.")
    
    def find_canonical_form(board, encoding_func):
        return board, {'rotation': 0, 'reflected': False, 'original_to_canonical': 0}
    
    def board_to_encoding_key(board, encoding_func):
        return hash(str(board.piece_map()))

# Cache para canonical forms (evita recalcular)
_canonical_cache = {}

def get_canonical_key(board, encoding_func):
    """Obtener clave canónica con cache."""
    # Crear clave única para el tablero
    board_key = (board.turn, frozenset(board.piece_map().items()))
    
    if board_key in _canonical_cache:
        return _canonical_cache[board_key]
    
    # Calcular forma canónica
    canonical_board, _ = find_canonical_form(board, encoding_func)
    key = board_to_encoding_key(canonical_board, encoding_func)
    
    # Cachear
    _canonical_cache[board_key] = key
    return key

def process_chunk_canonical(args):
    """
    Procesar un chunk de combinaciones con canonical forms.
    Similar a process_chunk en generate_datasets_parallel.py pero con canonical forms.
    """
    chunk_idx, square_combinations, white_pieces, black_pieces, syzygy_path, \
    compact, relative, use_move_distance, progress_queue = args
    
    try:
        tablebase = chess.syzygy.open_tablebase(syzygy_path)
    except Exception as e:
        print(f"Chunk {chunk_idx}: Failed to open tablebase: {e}")
        return chunk_idx, [], [], [], [], 0
    
    positions = []
    labels_wdl = []
    labels_dtz = []
    canonical_keys = []
    valid_count = 0
    
    # Encoding function para canonical forms
    def encoding_func(board):
        return encode_board(board, compact=compact, relative=relative, 
                           use_move_distance=use_move_distance)
    
    for squares in square_combinations:
        board = chess.Board()
        board.clear_board()
        
        # Colocar piezas blancas
        for i, (piece, square) in enumerate(zip(white_pieces, squares[:len(white_pieces)])):
            board.set_piece_at(square, piece)
        
        # Colocar piezas negras
        for i, (piece, square) in enumerate(zip(black_pieces, squares[len(white_pieces):])):
            board.set_piece_at(square, piece)
        
        # Probar ambos turnos
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            
            if board.is_valid():
                try:
                    wdl = tablebase.probe_wdl(board)
                    dtz = tablebase.probe_dtz(board)
                    
                    # Obtener clave canónica
                    canonical_key = get_canonical_key(board, encoding_func)
                    
                    # Solo guardar si es la primera vez que vemos esta clave
                    # (en este chunk - la deduplicación final se hará después)
                    
                    encoding = encode_board(board, compact=compact, relative=relative,
                                           use_move_distance=use_move_distance)
                    
                    positions.append(encoding)
                    labels_wdl.append(wdl)
                    labels_dtz.append(dtz)
                    canonical_keys.append(canonical_key)
                    valid_count += 1
                    
                except Exception:
                    pass
    
    tablebase.close()
    
    if progress_queue:
        progress_queue.put((chunk_idx, valid_count))
    
    return chunk_idx, positions, labels_wdl, labels_dtz, canonical_keys, valid_count

def generate_dataset_canonical(syzygy_path: str, output_dir: str, config: str,
                              compact: bool = True, relative: bool = False,
                              use_move_distance: bool = False,
                              num_workers: int = None, chunk_size: int = 10000):
    """
    Generar dataset con canonical forms.
    """
    print("="*60)
    print("PARALLEL DATASET GENERATOR WITH CANONICAL FORMS")
    print("="*60)
    
    if not os.path.exists(syzygy_path):
        raise ValueError(f"Syzygy path {syzygy_path} not found.")
    
    # Determinar piezas basado en config
    if 'v' in config:
        white_side, black_side = config.split('v')
    else:
        white_side = config[:-1]
        black_side = config[-1]
    
    print(f"Generating dataset for {white_side} vs {black_side}...")
    print(f"  White pieces: {white_side}")
    print(f"  Black pieces: {black_side}")
    print(f"  Using canonical forms (8 symmetries)")
    
    # Convertir símbolos a piezas
    def symbols_to_pieces(symbols):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in symbols]
    
    w_pieces = symbols_to_pieces(white_side)
    b_pieces = symbols_to_pieces(black_side)
    for p in b_pieces:
        p.color = chess.BLACK
    
    total_pieces = len(w_pieces) + len(b_pieces)
    print(f"  Total pieces: {total_pieces}")
    
    # Calcular combinaciones totales
    total_combinations = 64 ** total_pieces
    print(f"  Total combinations: {total_combinations:,}")
    
    # Determinar número de workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    print(f"  Workers: {num_workers}")
    print(f"  Chunk size: {chunk_size:,}")
    
    # Crear chunks
    import math
    total_chunks = math.ceil(total_combinations / chunk_size)
    print(f"  Total chunks: {total_chunks}")
    
    # Función para generar combinaciones de un chunk
    def generate_chunk_combinations(chunk_idx):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_combinations)
        
        combinations = []
        for idx in range(start, end):
            # Convertir índice a combinación (simplificado)
            squares = []
            n = idx
            for _ in range(total_pieces):
                squares.append(n % 64)
                n //= 64
            
            # Verificar que no haya duplicados (piezas en la misma casilla)
            if len(set(squares)) == total_pieces:
                combinations.append(tuple(squares))
        
        return combinations
    
    print("\nProcessing with canonical forms and incremental deduplication...")
    
    # Usar Manager para progress queue
    with Manager() as manager:
        progress_queue = manager.Queue()
        
        # Preparar argumentos para workers
        chunk_args = []
        for chunk_idx in range(total_chunks):
            square_combinations = generate_chunk_combinations(chunk_idx)
            if square_combinations:
                chunk_args.append((
                    chunk_idx, square_combinations, w_pieces, b_pieces,
                    syzygy_path, compact, relative, use_move_distance,
                    progress_queue
                ))
        
        start_time = time.time()
        positions_all = []
        wdl_all = []
        dtz_all = []
        canonical_keys_all = []
        total_valid = 0
        
        # Diccionario para deduplicación por clave canónica
        canonical_dict = {}
        
        # Procesar chunks
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk_canonical, args): args[0] 
                      for args in chunk_args}
            
            completed = 0
            for future in as_completed(futures):
                chunk_idx = futures[future]
                
                try:
                    result = future.result()
                    chunk_idx, positions, wdl, dtz, canonical_keys, valid_count = result
                    
                    # Procesar resultados del chunk
                    for i in range(len(positions)):
                        key = canonical_keys[i]
                        
                        # Solo guardar si es la primera vez que vemos esta clave canónica
                        if key not in canonical_dict:
                            canonical_dict[key] = len(positions_all)
                            positions_all.append(positions[i])
                            wdl_all.append(wdl[i])
                            dtz_all.append(dtz[i])
                    
                    total_valid += valid_count
                    completed += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = total_valid / elapsed
                        eta = (total_chunks - completed) * (elapsed / completed) if completed > 0 else 0
                        
                        print(f"Progress: {completed}/{total_chunks} chunks ({completed/total_chunks*100:.1f}%) | "
                              f"Positions: {total_valid:,} | Unique: {len(positions_all):,} | "
                              f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                              f"ETA: {timedelta(seconds=int(eta))} | "
                              f"Reduction: {1 - len(positions_all)/total_valid if total_valid > 0 else 0:.1%}")
                    
                except Exception as e:
                    print(f"Chunk {chunk_idx} failed: {e}")
        
        elapsed = time.time() - start_time
        
        print(f"\nFound {total_valid:,} valid positions.")
        print(f"After canonical forms deduplication: {len(positions_all):,} unique positions.")
        print(f"Reduction: {1 - len(positions_all)/total_valid:.1%}")
        
        if len(positions_all) == 0:
            print("No valid positions found!")
            return
        
        # Convertir a arrays numpy
        x_array = np.array(positions_all, dtype=np.float32)
        wdl_array = np.array(wdl_all, dtype=np.int8)
        dtz_array = np.array(dtz_all, dtype=np.int16)
        
        print(f"\nFinal dataset shape: {x_array.shape}")
        print(f"  Samples: {x_array.shape[0]:,}")
        print(f"  Dimensions per sample: {x_array.shape[1]}")
        
        # Guardar dataset
        output_path = os.path.join(output_dir, f"{config}_canonical.npz")
        np.savez_compressed(output_path, x=x_array, wdl=wdl_array, dtz=dtz_array)
        
        print(f"\nSaved to {output_path}")
        print(f"Total time: {timedelta(seconds=int(elapsed))}")
        print(f"Speed: {total_valid/elapsed:.0f} positions/second")
        
        # Guardar metadata
        metadata = {
            'config': config,
            'total_valid': total_valid,
            'unique_positions': len(positions_all),
            'reduction': 1 - len(positions_all)/total_valid,
            'dimensions': x_array.shape[1],
            'encoding': 'relative' if relative else 'compact',
            'use_move_distance': use_move_distance,
            'canonical_forms': True
        }
        
        metadata_path = os.path.join(output_dir, f"{config}_canonical_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadata saved to {metadata_path}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate dataset with canonical forms")
    parser.add_argument("--syzygy", type=str, default="syzygy",
                       help="Path to Syzygy tablebase directory")
    parser.add_argument("--data", type=str, default="data",
                       help="Output directory for datasets")
    parser.add_argument("--config", type=str, required=True,
                       help="Endgame configuration (e.g., KQvK, KRRvK)")
    parser.add_argument("--compact", action="store_true", default=True,
                       help="Use compact encoding (default: True)")
    parser.add_argument("--full", action="store_true",
                       help="Use full encoding (disables compact)")
    parser.add_argument("--relative", action="store_true",
                       help="Use relative/geometric encoding")
    parser.add_argument("--move-distance", action="store_true",
                       help="Include piece-specific move distance (encoding v2)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count, max 8)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Number of combinations per chunk (default: 10000)")
    
    args = parser.parse_args()
    
    compact = args.compact and not args.full
    
    generate_dataset_canonical(
        syzygy_path=args.syzygy,
        output_dir=args.data,
        config=args.config,
        compact=compact,
        relative=args.relative,
        use_move_distance=args.move_distance,
        num_workers=args.workers,
        chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main()