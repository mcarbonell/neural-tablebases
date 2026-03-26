import ctypes
import os
import torch
import numpy as np

# Define the C struct to match Rust
class GnnFeatureStruct(ctypes.Structure):
    _fields_ = [
        ("piece_ids", ctypes.c_byte * 64),
        ("white_atk_counts", ctypes.c_byte * 64),
        ("black_atk_counts", ctypes.c_byte * 64),
        ("attacker_bitmasks", ctypes.c_ubyte * 64),
        ("flags", ctypes.c_ubyte * 64),
        ("edge_count", ctypes.c_uint32),
        ("edges", ctypes.c_uint16 * 1024),
    ]

class RustGnnEngine:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Default location
            ext = ".dll" if os.name == "nt" else ".so"
            lib_path = os.path.join(os.path.dirname(__file__), "rust_movegen", "target", "release", f"rust_gnn_engine{ext}")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Rust GNN engine not found at {lib_path}. Please run 'cargo build --release' in src/rust_movegen.")
        
        self.lib = ctypes.CDLL(lib_path)
        self.lib.generate_gnn_features.argtypes = [ctypes.c_char_p, ctypes.POINTER(GnnFeatureStruct)]
        self.lib.generate_gnn_features.restype = ctypes.c_int32

    def get_features(self, fen):
        out = GnnFeatureStruct()
        res = self.lib.generate_gnn_features(fen.encode('ascii'), ctypes.byref(out))
        if res != 0:
            raise Exception(f"Rust engine error: {res}")
        
        # Convert to Tensors
        piece_ids = torch.from_numpy(np.array(out.piece_ids, dtype=np.int64))
        
        # Tactical features: 
        # [white_atks, black_atks, bitmask, flags] -> total 4 features per square?
        # Actually our model ChessGNN expects tactical_features=8 (default)
        # We can stack them: white_atks, black_atks, flags, bitmask
        tactical = np.zeros((64, 4), dtype=np.float32)
        tactical[:, 0] = np.array(out.white_atk_counts)
        tactical[:, 1] = np.array(out.black_atk_counts)
        tactical[:, 2] = np.array(out.flags)
        tactical[:, 3] = np.array(out.attacker_bitmasks)
        tactical_tensor = torch.from_numpy(tactical)

        adj = torch.zeros((64, 64, 16), dtype=torch.float32)
        for i in range(out.edge_count):
            encoded = out.edges[i]
            e_type = (encoded >> 12) & 0xF
            src = (encoded >> 6) & 0x3F
            dst = encoded & 0x3F
            adj[src, dst, e_type] = 1.0
        
        return piece_ids, tactical_tensor, adj

    def get_raw_features(self, fen):
        """
        Returns the raw features from the Rust engine.
        Useful for dataset generation (much faster and more compact).
        """
        out = GnnFeatureStruct()
        res = self.lib.generate_gnn_features(fen.encode('ascii'), ctypes.byref(out))
        if res != 0:
            raise Exception(f"Rust engine error: {res}")
        
        # Convert to numpy directly
        piece_ids = np.array(out.piece_ids, dtype=np.int8)
        
        # Stack tactical features: [white_atks, black_atks, flags, bitmask]
        tactical = np.zeros((64, 4), dtype=np.uint8)
        tactical[:, 0] = np.array(out.white_atk_counts)
        tactical[:, 1] = np.array(out.black_atk_counts)
        tactical[:, 2] = np.array(out.flags)
        tactical[:, 3] = np.array(out.attacker_bitmasks)
        
        # Raw edges and count
        edges = np.array(out.edges, dtype=np.uint16)
        edge_count = out.edge_count
        
        return piece_ids, tactical, edges, int(edge_count)

# Test usage
if __name__ == "__main__":
    try:
        engine = RustGnnEngine()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p_ids, tac, adj = engine.get_features(fen)
        print(f"Loaded FEN: {fen}")
        print(f"Piece IDs: {p_ids.shape}")
        print(f"Tactical features: {tac.shape}")
        print(f"Adjacency: {adj.shape}")
        print(f"Edge count (Total): {torch.sum(adj)}")
    except Exception as e:
        print(f"Error: {e}")
