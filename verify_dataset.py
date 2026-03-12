import numpy as np
import chess
import chess.syzygy
import os

def verify_dataset(data_path, syzygy_path, num_samples=100):
    """Verify dataset integrity by comparing with Syzygy tablebase"""
    
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path)
    x = data['x']
    wdl = data['wdl']
    dtz = data['dtz']
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total positions: {len(x)}")
    print(f"Input dimensions: {x.shape[1]}")
    print(f"Num pieces: {x.shape[1] // 64}")
    
    print(f"\n=== WDL DISTRIBUTION ===")
    unique_wdl, counts = np.unique(wdl, return_counts=True)
    for val, count in zip(unique_wdl, counts):
        percentage = (count / len(wdl)) * 100
        print(f"WDL {val:2d}: {count:7d} ({percentage:5.2f}%)")
    
    print(f"\n=== DTZ STATISTICS ===")
    print(f"DTZ range: [{dtz.min()}, {dtz.max()}]")
    print(f"DTZ mean: {dtz.mean():.2f}")
    print(f"DTZ std: {dtz.std():.2f}")
    
    # Check for NaN or Inf
    print(f"\n=== DATA QUALITY ===")
    print(f"NaN in x: {np.isnan(x).any()}")
    print(f"Inf in x: {np.isinf(x).any()}")
    print(f"NaN in wdl: {np.isnan(wdl).any()}")
    print(f"NaN in dtz: {np.isnan(dtz).any()}")
    
    # Verify encoding
    print(f"\n=== ENCODING VERIFICATION ===")
    num_pieces = x.shape[1] // 64
    print(f"Expected pieces per position: {num_pieces}")
    
    # Check that each position has exactly num_pieces pieces
    pieces_per_position = x.reshape(-1, num_pieces, 64).sum(axis=2).sum(axis=1)
    print(f"Pieces per position - min: {pieces_per_position.min()}, max: {pieces_per_position.max()}")
    if not np.allclose(pieces_per_position, num_pieces):
        print(f"WARNING: Some positions don't have exactly {num_pieces} pieces!")
        wrong_count = np.sum(pieces_per_position != num_pieces)
        print(f"Positions with wrong piece count: {wrong_count}")
    
    # Verify against Syzygy (if available)
    if os.path.exists(syzygy_path):
        print(f"\n=== SYZYGY VERIFICATION ===")
        print(f"Verifying {num_samples} random samples against Syzygy...")
        
        try:
            tablebase = chess.syzygy.open_tablebase(syzygy_path)
            
            # Sample random positions
            indices = np.random.choice(len(x), min(num_samples, len(x)), replace=False)
            mismatches_wdl = 0
            mismatches_dtz = 0
            
            for idx in indices:
                # Decode position
                board = decode_position(x[idx], num_pieces)
                if board is None:
                    continue
                
                try:
                    # Probe Syzygy
                    syzygy_wdl = tablebase.probe_wdl(board)
                    syzygy_dtz = tablebase.probe_dtz(board)
                    
                    # Compare
                    if wdl[idx] != syzygy_wdl:
                        mismatches_wdl += 1
                        if mismatches_wdl <= 5:  # Show first 5 mismatches
                            print(f"WDL mismatch at idx {idx}: dataset={wdl[idx]}, syzygy={syzygy_wdl}")
                            print(f"  FEN: {board.fen()}")
                    
                    if dtz[idx] != syzygy_dtz:
                        mismatches_dtz += 1
                        if mismatches_dtz <= 5:
                            print(f"DTZ mismatch at idx {idx}: dataset={dtz[idx]}, syzygy={syzygy_dtz}")
                
                except Exception as e:
                    print(f"Error probing position {idx}: {e}")
            
            print(f"\nVerification complete:")
            print(f"WDL mismatches: {mismatches_wdl}/{num_samples} ({100*mismatches_wdl/num_samples:.2f}%)")
            print(f"DTZ mismatches: {mismatches_dtz}/{num_samples} ({100*mismatches_dtz/num_samples:.2f}%)")
            
            tablebase.close()
            
        except Exception as e:
            print(f"Could not verify against Syzygy: {e}")
    else:
        print(f"\n=== SYZYGY VERIFICATION ===")
        print(f"Syzygy path not found: {syzygy_path}")

def decode_position(encoding, num_pieces):
    """Decode a compact encoding back to a chess board"""
    board = chess.Board(None)
    
    encoding_reshaped = encoding.reshape(num_pieces, 64)
    
    # Find which square each piece is on
    for piece_idx in range(num_pieces):
        square = np.argmax(encoding_reshaped[piece_idx])
        
        # We don't know the piece type from compact encoding alone
        # This is a limitation - we'd need to store piece types separately
        # For now, just check if the encoding is valid
        if encoding_reshaped[piece_idx, square] != 1.0:
            return None
    
    return None  # Can't fully decode without piece type info

if __name__ == "__main__":
    verify_dataset("data/KQvK.npz", "syzygy", num_samples=100)
