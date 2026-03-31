# V8 Evolution: GNN Neural Search Architecture (Fusion V8.1)

## 🚀 The GNN Breakthrough
The transition from V7 (MLP) to V8 (Graph Neural Network) represents a shift from "Feature Engineering" to "Topological Reasoning". 

### Prototype Results (Baseline V8.0)
- **Status**: **Success** (99.83% Accuracy on KQvK smoke test).
- **Core Finding**: Graphs learn tactical relations orders of magnitude faster than flat vectors.

---

## 🏗️ Evolution 8.1: Rust-Neural Fusion
Instead of a generic graph, V8.1 leverages a high-performance Rust movegen (25M n/s) to provide a "Tactical Backbone" for the neural network.

### Node Features (Input $X$)
Instead of raw IDs, each square node now contains:
- **Piece Identity**: Type and Color.
- **Tactical Counters**: Number of attackers per side and bitmask of piece types (from Rust x88 engine).
- **Status Flags**: Is Pinned, Is Checked, King Oxygen.

### Multi-Channel Adjacency (The "Cables")
The network processes 4 distinct types of tactical flow simultaneously:
1. **Mobility**: Strictly legal moves.
2. **Influence**: All squares attacked/defended (ignoring pins/legality).
3. **X-Ray**: Strategic vision through pieces (essential for pin/skewer detection).
4. **King Pressure**: Direct signals to squares adjacent to the enemy king.

---

## 🛠️ Performance & Scalability
- **Compute**: The Radeon 780 iGPU handles the GNN inference via DirectML.
- **Memory**: With 64GB RAM, we can cache millions of board states and adjacency masks.
- **Speed**: The Rust engine is so fast (25M n/s) that graph generation happens "Just-In-Time" during training without bottlenecking the GPU.

---

## 📅 Roadmap V8.1
- [x] **V8.0 Prototype**: Verification of GNN potential (99.8% hit).
- [ ] **Bridge Implementation**: Connect Python to the Rust/WASM Bitboard engine.
- [ ] **Universal Sharding V8**: Generate the 40M universe with multi-channel adjacency.
- [ ] **GNN-Fusion Training**: Target 99.9%+ accuracy by simulating tactical depth inside the network.
