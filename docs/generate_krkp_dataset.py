import chess
import chess.syzygy
import numpy as np
import os
import argparse

def generate_krkp_dataset(syzygy_path, output_file):
    """
    Generates a dataset for King + Rook vs King + Pawn.
    """
    if not os.path.exists(syzygy_path):
        print(f"Error: Syzygy path {syzygy_path} not found.")
        print("Please download the 4-piece Syzygy tablebases (e.g., from https://tablebase.lichess.ovh/tables/syzygy/345/)")
        return

    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Pieces: WK (White King), WR (White Rook), BK (Black King), BP (Black Pawn)
    # Total combinations: 64^4 = 16.7M (before legality checks)
    
    positions = []
    labels_wdl = [] # -1 (Loss), 0 (Draw), 1 (Win)
    labels_dtz = [] # Distance to zeroing-move
    
    print("Starting position generation for KRvKP...")
    
    count = 0
    valid_count = 0
    
    # We'll fix White to move for this dataset generation
    for wk in range(64):
        for wr in range(64):
            if wr == wk: continue
            for bk in range(64):
                if bk == wk or bk == wr: continue
                # Pawn cannot be on 1st or 8th rank
                for bp in range(8, 56):
                    if bp == wk or bp == wr or bp == bk: continue
                    
                    # Create board
                    board = chess.Board(None)
                    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
                    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                    board.set_piece_at(bp, chess.Piece(chess.PAWN, chess.BLACK))
                    board.turn = chess.WHITE
                    
                    # Basic legality check
                    if board.is_valid():
                        try:
                            wdl = tablebase.probe_wdl(board)
                            dtz = tablebase.probe_dtz(board)
                            
                            # Convert to format: [wk, wr, bk, bp]
                            positions.append([wk, wr, bk, bp])
                            labels_wdl.append(wdl)
                            labels_dtz.append(dtz)
                            valid_count += 1
                        except Exception:
                            # Position might not be in the loaded tablebases (rare for 4-piece if path is correct)
                            pass
                    
                    count += 1
                    if count % 100000 == 0:
                        print(f"Processed {count} positions... Found {valid_count} valid.")

    print(f"Generation complete. Found {valid_count} valid positions.")
    
    # Save as compressed NumPy array
    np.savez_compressed(output_file, 
                        positions=np.array(positions, dtype=np.uint8),
                        wdl=np.array(labels_wdl, dtype=np.int8),
                        dtz=np.array(labels_dtz, dtype=np.int16))
    
    print(f"Dataset saved to {output_file}")
    tablebase.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KRvKP Chess Dataset")
    parser.add_argument("--syzygy", type=str, required=True, help="Path to Syzygy 4-piece tablebases")
    parser.add_argument("--output", type=str, default="dataset_krkp.npz", help="Output .npz file")
    
    args = parser.parse_args()
    generate_krkp_dataset(args.syzygy, args.output)
