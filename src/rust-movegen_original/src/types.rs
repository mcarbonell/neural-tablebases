use wasm_bindgen::prelude::*;

// Side to move
pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

// Piece types
pub const EMPTY: u8 = 0;
pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

// Pieces (Color | PieceType)
pub const WP: u8 = 1;
pub const WN: u8 = 2;
pub const WB: u8 = 3;
pub const WR: u8 = 4;
pub const WQ: u8 = 5;
pub const WK: u8 = 6;
pub const BP: u8 = 9;
pub const BN: u8 = 10;
pub const BB: u8 = 11;
pub const BR: u8 = 12;
pub const BQ: u8 = 13;
pub const BK: u8 = 14;

// Board squares (x88)
pub const A1: i32 = 0;   pub const B1: i32 = 1;   pub const C1: i32 = 2;   pub const D1: i32 = 3;
pub const E1: i32 = 4;   pub const F1: i32 = 5;   pub const G1: i32 = 6;   pub const H1: i32 = 7;
pub const A2: i32 = 16;  pub const H2: i32 = 23;
pub const A7: i32 = 96;  pub const H7: i32 = 103;
pub const A8: i32 = 112; pub const B8: i32 = 113; pub const C8: i32 = 114; pub const D8: i32 = 115;
pub const E8: i32 = 116; pub const F8: i32 = 117; pub const G8: i32 = 118; pub const H8: i32 = 119;

// Castling rights
pub const WHITE_CASTLE_K: u8 = 1;
pub const WHITE_CASTLE_Q: u8 = 2;
pub const BLACK_CASTLE_K: u8 = 4;
pub const BLACK_CASTLE_Q: u8 = 8;

// Move masks (16-bit)
pub const MASK_CAPTURE: u16 = 1;
pub const MASK_PROMOTION: u16 = 1 << 1;
pub const MASK_PAWN_MOVE: u16 = 1 << 2;
pub const MASK_EP: u16 = 1 << 3;
pub const MASK_CASTLING: u16 = 1 << 4;
pub const MASK_CHECK: u16 = 1 << 5;
pub const MASK_DOUBLE_CHECK: u16 = 1 << 6;
pub const MASK_DISCOVER_CHECK: u16 = 1 << 7;
pub const MASK_SAFE: u16 = 1 << 8;
pub const MASK_PROTECTED: u16 = 1 << 9;
pub const MASK_INSECURE: u16 = 1 << 10;
pub const MASK_HANGING: u16 = 1 << 11;
pub const MASK_GOOD_CAPTURE: u16 = 1 << 12;
pub const MASK_FREE_CAPTURE: u16 = 1 << 13;
pub const MASK_WINNING_CAPTURE: u16 = 1 << 14;
pub const MASK_PAWN_2: u16 = 1 << 15;

// Compatibility aliases for x88 engine
pub const MASK_DISCOVERCHECK: u16 = MASK_DISCOVER_CHECK;
pub const MASK_FREECAPTURE: u16 = MASK_FREE_CAPTURE;
pub const MASK_WINNINGCAPTURE: u16 = MASK_WINNING_CAPTURE;
pub const MASK_PAWNMOVE: u16 = MASK_PAWN_MOVE;

// Piece check bits
pub const CHECKBIT_PAWN: u8 = 1;
pub const CHECKBIT_KNIGHT: u8 = 1 << 1;
pub const CHECKBIT_BISHOP: u8 = 1 << 2;
pub const CHECKBIT_ROOK: u8 = 1 << 3;
pub const CHECKBIT_QUEEN: u8 = CHECKBIT_BISHOP | CHECKBIT_ROOK;
pub const CHECKBIT_KING: u8 = 1 << 5;

// Attack bits
pub const ATTACKBIT_PAWN: u8 = 1;
pub const ATTACKBIT_KNIGHT: u8 = 1 << 1;
pub const ATTACKBIT_BISHOP: u8 = 1 << 2;
pub const ATTACKBIT_ROOK: u8 = 1 << 3;
pub const ATTACKBIT_QUEEN: u8 = 1 << 4;
pub const ATTACKBIT_KING: u8 = 1 << 5;

#[wasm_bindgen]
#[derive(Copy, Clone, Default)]
pub struct Move {
    pub from: i32,
    pub to: i32,
    pub moving: u8,
    pub captured: u8,
    pub promoted: u8,
    pub mask: u16,
}

#[derive(Copy, Clone, Default)]
pub struct HistoryEntry {
    pub from: i32,
    pub to: i32,
    pub captured: u8,
    pub promoted: u8,
    pub c50: u32,
    pub cr: u8,
    pub ep: i32,
}

pub fn pcolor(piece: u8) -> i32 {
    if piece == 0 { -1 } else { ((piece >> 3) & 1) as i32 }
}

pub fn ptype(piece: u8) -> u8 {
    piece & 7
}

pub fn makepiece(color: usize, p_type: u8) -> u8 {
    ((color as u8) << 3) | p_type
}

pub fn checkbit(p_type: u8) -> u8 {
    match p_type {
        PAWN => CHECKBIT_PAWN,
        KNIGHT => CHECKBIT_KNIGHT,
        BISHOP => CHECKBIT_BISHOP,
        ROOK => CHECKBIT_ROOK,
        QUEEN => CHECKBIT_QUEEN,
        KING => CHECKBIT_KING,
        _ => 0,
    }
}

pub fn attackbit(p_type: u8) -> u8 {
    match p_type {
        PAWN => ATTACKBIT_PAWN,
        KNIGHT => ATTACKBIT_KNIGHT,
        BISHOP => ATTACKBIT_BISHOP,
        ROOK => ATTACKBIT_ROOK,
        QUEEN => ATTACKBIT_QUEEN,
        KING => ATTACKBIT_KING,
        _ => 0,
    }
}

// X88 Helper functions
#[inline(always)]
pub fn valid_square(sq: i32) -> bool {
    (sq & 0x88) == 0
}

#[inline(always)]
pub fn square(f: i32, r: i32) -> i32 {
    (r << 4) | f
}

#[inline(always)]
pub fn rank(sq: i32) -> i32 {
    sq >> 4
}

#[inline(always)]
pub fn get_file(sq: i32) -> i32 {
    sq & 7
}

pub const FILE_A: i32 = 0; pub const FILE_B: i32 = 1; pub const FILE_C: i32 = 2; pub const FILE_D: i32 = 3;
pub const FILE_E: i32 = 4; pub const FILE_F: i32 = 5; pub const FILE_G: i32 = 6; pub const FILE_H: i32 = 7;

pub const RANK_1: i32 = 0; pub const RANK_2: i32 = 1; pub const RANK_3: i32 = 2; pub const RANK_4: i32 = 3;
pub const RANK_5: i32 = 4; pub const RANK_6: i32 = 5; pub const RANK_7: i32 = 6; pub const RANK_8: i32 = 7;

pub const MAXPLY: usize = 128;

pub static OFFSETS_KNIGHT: [i32; 8] = [14, 18, 31, 33, -14, -18, -31, -33];
pub static OFFSETS_BISHOP: [i32; 4] = [15, 17, -15, -17];
pub static OFFSETS_ROOK: [i32; 4] = [1, 16, -1, -16];
pub static OFFSETS_QUEEN_KING: [i32; 8] = [1, 16, -1, -16, 15, 17, -15, -17];

pub static RAYS: [i8; 240] = [
    17, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 15, 0,
    0, 17, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 15, 0, 0,
    0, 0, 17, 0, 0, 0, 0, 16, 0, 0, 0, 0, 15, 0, 0, 0,
    0, 0, 0, 17, 0, 0, 0, 16, 0, 0, 0, 15, 0, 0, 0, 0,
    0, 0, 0, 0, 17, 0, 0, 16, 0, 0, 15, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 17, 0, 16, 0, 15, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 17, 16, 15, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, 0,
    0, 0, 0, 0, 0, 0, -15, -16, -17, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -15, 0, -16, 0, -17, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -15, 0, 0, -16, 0, 0, -17, 0, 0, 0, 0, 0,
    0, 0, 0, -15, 0, 0, 0, -16, 0, 0, 0, -17, 0, 0, 0, 0,
    0, 0, -15, 0, 0, 0, 0, -16, 0, 0, 0, 0, -17, 0, 0, 0,
    0, -15, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, -17, 0, 0,
    -15, 0, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, -17, 0
];

pub static PIECE_BY_OFFSET: [u8; 18] = {
    let mut p = [0u8; 18];
    p[1] = CHECKBIT_ROOK;
    p[16] = CHECKBIT_ROOK;
    p[15] = CHECKBIT_BISHOP;
    p[17] = CHECKBIT_BISHOP;
    p
};

// Bitboard constants
pub const UNIVERSE: u64 = 0xFFFFFFFFFFFFFFFF;
pub const BB_FILE_A: u64 = 0x0101010101010101;
pub const BB_FILE_B: u64 = 0x0202020202020202;
pub const BB_FILE_C: u64 = 0x0404040404040404;
pub const BB_FILE_D: u64 = 0x0808080808080808;
pub const BB_FILE_E: u64 = 0x1010101010101010;
pub const BB_FILE_F: u64 = 0x2020202020202020;
pub const BB_FILE_G: u64 = 0x4040404040404040;
pub const BB_FILE_H: u64 = 0x8080808080808080;

pub const BB_RANK_1: u64 = 0x00000000000000FF;
pub const BB_RANK_2: u64 = 0x000000000000FF00;
pub const BB_RANK_3: u64 = 0x0000000000FF0000;
pub const BB_RANK_4: u64 = 0x00000000FF000000;
pub const BB_RANK_5: u64 = 0x000000FF00000000;
pub const BB_RANK_6: u64 = 0x0000FF0000000000;
pub const BB_RANK_7: u64 = 0x00FF000000000000;
pub const BB_RANK_8: u64 = 0xFF00000000000000;

pub const NOT_A_FILE: u64 = !BB_FILE_A;
pub const NOT_H_FILE: u64 = !BB_FILE_H;
pub const NOT_AB_FILE: u64 = !(BB_FILE_A | BB_FILE_B);
pub const NOT_GH_FILE: u64 = !(BB_FILE_G | BB_FILE_H);

pub fn char_to_piece(c: char) -> u8 {
    match c {
        'P' => WP, 'N' => WN, 'B' => WB, 'R' => WR, 'Q' => WQ, 'K' => WK,
        'p' => BP, 'n' => BN, 'b' => BB, 'r' => BR, 'q' => BQ, 'k' => BK,
        _ => EMPTY,
    }
}

pub fn opposite(color: usize) -> usize {
    1 - color
}
