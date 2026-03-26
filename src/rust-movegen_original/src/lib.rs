use wasm_bindgen::prelude::*;
pub mod types;
pub mod board;
pub mod magic_tables;
pub mod bitboard_board;

pub use board::Board;
pub use bitboard_board::BitboardBoard;
use types::Move;
use types::*;

#[wasm_bindgen]
pub struct RustBoard {
    inner: Box<Board>,
}

#[wasm_bindgen]
impl RustBoard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RustBoard {
        RustBoard { inner: Box::new(Board::new()) }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[wasm_bindgen(js_name = loadFEN)]
    pub fn load_fen(&mut self, fen: &str) {
        self.inner.load_fen(fen);
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        self.inner.perft(depth)
    }

    pub fn generate_moves(&mut self) {
        self.inner.generate_moves();
    }

    pub fn move_count(&self) -> u32 {
        let ply = self.inner.history_ply;
        self.inner.move_count[ply] as u32
    }

    pub fn get_move_from(&self, index: u32) -> i32 {
        let ply = self.inner.history_ply;
        self.inner.moves[ply][index as usize].from
    }

    pub fn get_move_to(&self, index: u32) -> i32 {
        let ply = self.inner.history_ply;
        self.inner.moves[ply][index as usize].to
    }

    pub fn make_move(&mut self, index: u32) -> bool {
        let ply = self.inner.history_ply;
        let m = self.inner.moves[ply][index as usize];
        self.inner.make_move(m)
    }

    pub fn undo_move(&mut self) -> bool {
        self.inner.undo_move()
    }

    pub fn is_attacked(&self, sq: i32, by_side: usize) -> bool {
        self.inner.is_attacked(sq, by_side)
    }
}

#[wasm_bindgen]
pub struct RustBitboardBoard {
    inner: Box<BitboardBoard>,
}

#[wasm_bindgen]
impl RustBitboardBoard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RustBitboardBoard {
        RustBitboardBoard { inner: Box::new(BitboardBoard::new()) }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[wasm_bindgen(js_name = loadFEN)]
    pub fn load_fen(&mut self, fen: &str) {
        self.inner.load_fen(fen);
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        self.inner.perft(depth)
    }

    pub fn generate_moves(&mut self) {
        self.inner.generate_moves();
    }

    pub fn move_count(&self) -> u32 {
        self.inner.moves_cache.len() as u32
    }

    pub fn get_move_from(&self, index: u32) -> i32 {
        self.inner.moves_cache[index as usize].from
    }

    pub fn get_move_to(&self, index: u32) -> i32 {
        self.inner.moves_cache[index as usize].to
    }

    pub fn make_move(&mut self, index: u32) -> bool {
        let m = self.inner.moves_cache[index as usize];
        self.inner.make_move(m)
    }

    pub fn undo_move(&mut self) -> bool {
        self.inner.undo_move()
    }

    pub fn is_attacked(&self, sq: usize, by_side: usize) -> bool {
        self.inner.is_attacked(sq, by_side)
    }
}
