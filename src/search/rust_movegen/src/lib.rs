use std::ffi::CStr;
use std::os::raw::c_char;

pub mod types;
pub mod magic_tables;
pub mod bitboard_board;
pub mod x88_board;

pub use x88_board::X88Board;
use types::*;

#[repr(C)]
pub struct GnnFeatureStruct {
    pub piece_ids: [i8; 64],
    pub white_atk_counts: [i8; 64],
    pub black_atk_counts: [i8; 64],
    pub attacker_bitmasks: [u8; 64],
    pub flags: [u8; 64],
    pub edge_count: u32,
    pub edges: [u16; 1024], // (type << 12) | (src << 6) | dst
}

#[no_mangle]
pub extern "C" fn generate_gnn_features(fen_ptr: *const c_char, out: *mut GnnFeatureStruct) -> i32 {
    if fen_ptr.is_null() || out.is_null() {
        return -1;
    }

    let c_str = unsafe { CStr::from_ptr(fen_ptr) };
    let fen = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let mut board = X88Board::new();
    board.load_fen(fen);
    
    let out_ref = unsafe { &mut *out };
    board.fill_gnn_features(out_ref);

    0
}
