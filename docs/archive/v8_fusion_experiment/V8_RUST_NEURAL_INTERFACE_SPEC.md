# V8.1 Rust-Neural Interface Specification

This document defines the data protocol for the bridge between the Rust Move Generator and the Python GNN Training Pipeline.

## 1. Unified Board Representation (Input Struct)
The bridge should pass a packed struct for each position.

| Feature | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **Piece IDs** | `i8` | 64 | 0 (Empty), 1-6 (White P-K), 7-12 (Black P-K) |
| **Attacker Counts (W)** | `i8` | 64 | Number of white pieces attacking each square |
| **Attacker Counts (B)** | `i8` | 64 | Number of black pieces attacking each square |
| **Attacker Bitmask** | `u8` | 64 | Bitmask of piece types (bit 0=P, 1=N, ...) attacking the square |
| **Flags** | `u8` | 64 | Bit 0: Pinned, Bit 1: Checked, Bit 2: King Oxygen |

## 2. Multi-Channel Adjacency List
To avoid dense 64x64x4 tensors, Rust should return a sparse list of edges.

### Edge Format: `(u16)`
Each edge is encoded as `(type << 12) | (src << 6) | dst`.
- **Bits 0-5**: Destination Square (0-63)
- **Bits 6-11**: Source Square (0-63)
- **Bits 12-15**: Edge Type (0: Move, 1: Influence, 2: X-Ray, 3: KingControl)

---

## 3. Python Integration (C-ABI)
The Rust library should expose a single function:
```rust
pub extern "C" fn generate_gnn_features(
    fen: *const c_char, 
    features: *mut GnnFeatureStruct
) -> i32;
```

This allows Python's `DataLoader` to call the Rust movegen at 25M pos/s, ensuring the GPU is always saturated with fresh tactical graphs.
