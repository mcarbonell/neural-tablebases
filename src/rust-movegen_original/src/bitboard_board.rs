use wasm_bindgen::prelude::*;
use crate::types::*;
use crate::magic_tables::*;

#[wasm_bindgen]
pub struct BitboardBoard {
    pub(crate) stm: usize,
    pub(crate) counter50: u32,
    pub(crate) castling_rights: u8,
    pub(crate) ep_square: i32,
    pub(crate) move_number: u32,
    pub(crate) move_half_number: u32,

    pub(crate) pieces: [[u64; 7]; 2],
    pub(crate) occ: [u64; 3],
    
    // We still keep a moves cache for the JS Wrapper to avoid frequent allocations
    pub(crate) moves_cache: Vec<Move>,
    pub(crate) history: Vec<HistoryEntry>,
}

#[wasm_bindgen]
impl BitboardBoard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> BitboardBoard {
        BitboardBoard {
            stm: WHITE,
            counter50: 0,
            castling_rights: 0,
            ep_square: -1,
            move_number: 1,
            move_half_number: 1,
            pieces: [[0; 7]; 2],
            occ: [0; 3],
            moves_cache: Vec::with_capacity(256),
            history: Vec::with_capacity(128),
        }
    }

    pub fn reset(&mut self) {
        self.stm = WHITE;
        self.counter50 = 0;
        self.castling_rights = 0;
        self.ep_square = -1;
        self.move_number = 1;
        self.move_half_number = 1;
        self.pieces = [[0; 7]; 2];
        self.occ = [0; 3];
        self.moves_cache.clear();
        self.history.clear();
    }

    pub fn load_fen(&mut self, fen: &str) {
        self.reset();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() { return; }

        let mut r = 7;
        let mut f = 0;
        for c in parts[0].chars() {
            if c == '/' { r -= 1; f = 0; }
            else if c.is_digit(10) { f += c.to_digit(10).unwrap() as i32; }
            else {
                let piece = char_to_piece(c);
                let color = if pcolor(piece) == 0 { WHITE } else { BLACK };
                let p_type = ptype(piece) as usize;
                let bit = 1u64 << (r * 8 + f);
                self.pieces[color][0] |= bit;
                self.pieces[color][p_type] |= bit;
                f += 1;
            }
        }
        self.update_occ();

        if parts.len() > 1 { self.stm = if parts[1] == "w" { WHITE } else { BLACK }; }
        if parts.len() > 2 {
            for c in parts[2].chars() {
                match c {
                    'K' => self.castling_rights |= WHITE_CASTLE_K,
                    'Q' => self.castling_rights |= WHITE_CASTLE_Q,
                    'k' => self.castling_rights |= BLACK_CASTLE_K,
                    'q' => self.castling_rights |= BLACK_CASTLE_Q,
                    _ => {}
                }
            }
        }
        if parts.len() > 3 && parts[3] != "-" {
            let f = (parts[3].chars().nth(0).unwrap() as i32) - 97;
            let r = (parts[3].chars().nth(1).unwrap() as i32) - 49;
            self.ep_square = r * 8 + f;
        }
        if parts.len() > 4 { self.counter50 = parts[4].parse().unwrap_or(0); }
        if parts.len() > 5 { 
            self.move_number = parts[5].parse().unwrap_or(1);
            self.move_half_number = self.move_number * 2 - (if self.stm == WHITE { 1 } else { 0 });
        }
    }

    fn update_occ(&mut self) {
        self.occ[0] = self.pieces[0][0];
        self.occ[1] = self.pieces[1][0];
        self.occ[2] = self.occ[0] | self.occ[1];
    }

    pub fn generate_moves_internal(&mut self) -> Vec<Move> {
        let mut pseudo_moves = Vec::with_capacity(128);
        let side = self.stm;
        let opp = 1 - side;
        let our_occ = self.occ[side];
        let total_occ = self.occ[2];
        let empty = !total_occ;

        // 1. King
        let king_sq = self.pieces[side][KING as usize].trailing_zeros() as usize;
        if king_sq < 64 {
            let atk = king_attacks(king_sq) & !our_occ;
            self.add_moves_to_vec(side, KING, king_sq, atk, &mut pseudo_moves);
        }
        // 2. Knight
        let mut knights = self.pieces[side][KNIGHT as usize];
        while knights != 0 {
            let sq = knights.trailing_zeros() as usize;
            knights &= knights - 1;
            let atk = knight_attacks(sq) & !our_occ;
            self.add_moves_to_vec(side, KNIGHT, sq, atk, &mut pseudo_moves);
        }
        // 3. Slider
        let mut bishops = self.pieces[side][BISHOP as usize] | self.pieces[side][QUEEN as usize];
        while bishops != 0 {
            let sq = bishops.trailing_zeros() as usize;
            bishops &= bishops - 1;
            let p_type = if (self.pieces[side][BISHOP as usize] & (1u64 << sq)) != 0 { BISHOP } else { QUEEN };
            let atk = bishop_attacks(sq, total_occ) & !our_occ;
            self.add_moves_to_vec(side, p_type, sq, atk, &mut pseudo_moves);
        }
        let mut rooks = self.pieces[side][ROOK as usize] | self.pieces[side][QUEEN as usize];
        while rooks != 0 {
            let sq = rooks.trailing_zeros() as usize;
            rooks &= rooks - 1;
            let p_type = if (self.pieces[side][ROOK as usize] & (1u64 << sq)) != 0 { ROOK } else { QUEEN };
            let atk = rook_attacks(sq, total_occ) & !our_occ;
            self.add_moves_to_vec(side, p_type, sq, atk, &mut pseudo_moves);
        }
        // 4. Pawn
        let pawns = self.pieces[side][PAWN as usize];
        let opp_occ = self.occ[opp];
        let bit_off = if side == WHITE { 8i32 } else { -8i32 };
        let s_push = if side == WHITE { (pawns << 8) & empty } else { (pawns >> 8) & empty };
        self.add_p_pushes_to_vec(side, bit_off, s_push, &mut pseudo_moves);
        let d_push = if side == WHITE { ((pawns & BB_RANK_2) << 8 & empty) << 8 & empty } 
                     else { ((pawns & BB_RANK_7) >> 8 & empty) >> 8 & empty };
        self.add_p_pushes_to_vec(side, bit_off * 2, d_push, &mut pseudo_moves);
        let l_cap = if side == WHITE { (pawns << 7) & opp_occ & NOT_H_FILE } else { (pawns >> 9) & opp_occ & NOT_H_FILE };
        self.add_p_caps_to_vec(side, if side == WHITE { 7 } else { -9 }, l_cap, &mut pseudo_moves);
        let r_cap = if side == WHITE { (pawns << 9) & opp_occ & NOT_A_FILE } else { (pawns >> 7) & opp_occ & NOT_A_FILE };
        self.add_p_caps_to_vec(side, if side == WHITE { 9 } else { -7 }, r_cap, &mut pseudo_moves);

        if self.ep_square != -1 {
            let to = self.ep_square as usize;
            let mut attackers = 0u64;
            if side == WHITE {
                if to >= 8 {
                    if (to % 8) > 0 { attackers |= (1u64 << (to - 9)) & pawns; }
                    if (to % 8) < 7 { attackers |= (1u64 << (to - 7)) & pawns; }
                }
            } else {
                if to < 56 {
                    if (to % 8) > 0 { attackers |= (1u64 << (to + 7)) & pawns; }
                    if (to % 8) < 7 { attackers |= (1u64 << (to + 9)) & pawns; }
                }
            }
            while attackers != 0 {
                let sq = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                pseudo_moves.push(Move {
                    from: sq as i32, to: self.ep_square, moving: makepiece(side, PAWN),
                    captured: makepiece(opp, PAWN), promoted: 0, mask: MASK_EP | MASK_CAPTURE,
                });
            }
        }

        // 5. Castle
        if side == WHITE {
            if (self.castling_rights & WHITE_CASTLE_K) != 0 && (total_occ & 0x60) == 0 {
                if !self.is_attacked(4, BLACK) && !self.is_attacked(5, BLACK) && !self.is_attacked(6, BLACK) {
                    pseudo_moves.push(Move { from: 4, to: 6, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & WHITE_CASTLE_Q) != 0 && (total_occ & 0x0E) == 0 {
                if !self.is_attacked(4, BLACK) && !self.is_attacked(3, BLACK) && !self.is_attacked(2, BLACK) {
                    pseudo_moves.push(Move { from: 4, to: 2, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        } else {
            if (self.castling_rights & BLACK_CASTLE_K) != 0 && (total_occ & (0x60u64 << 56)) == 0 {
                if !self.is_attacked(60, WHITE) && !self.is_attacked(61, WHITE) && !self.is_attacked(62, WHITE) {
                    pseudo_moves.push(Move { from: 60, to: 62, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & BLACK_CASTLE_Q) != 0 && (total_occ & (0x0Eu64 << 56)) == 0 {
                if !self.is_attacked(60, WHITE) && !self.is_attacked(59, WHITE) && !self.is_attacked(58, WHITE) {
                    pseudo_moves.push(Move { from: 60, to: 58, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        }

        let mut legal_moves = Vec::with_capacity(pseudo_moves.len());
        for m in pseudo_moves {
            if self.make_move(m) {
                legal_moves.push(m);
                self.undo_move();
            }
        }
        legal_moves
    }

    pub fn generate_moves(&mut self) {
        self.moves_cache = self.generate_moves_internal();
    }

    fn add_moves_to_vec(&self, side: usize, p_type: u8, from: usize, mut atk: u64, dest: &mut Vec<Move>) {
        let opp = 1 - side;
        let opp_occ = self.occ[opp];
        while atk != 0 {
            let to = atk.trailing_zeros() as usize;
            let to_bit = 1u64 << to;
            atk &= atk - 1;
            let mut mask = 0;
            let mut captured = 0;
            if (to_bit & opp_occ) != 0 {
                mask |= MASK_CAPTURE;
                captured = self.piece_at_sq(opp, to);
            }
            dest.push(Move { from: from as i32, to: to as i32, moving: makepiece(side, p_type), captured, promoted: 0, mask });
        }
    }

    fn add_p_pushes_to_vec(&self, side: usize, offset: i32, mut dests: u64, dest: &mut Vec<Move>) {
        while dests != 0 {
            let to = dests.trailing_zeros() as usize;
            dests &= dests - 1;
            let from = (to as i32) - offset;
            let to_bit = 1u64 << to;
            if (to_bit & (BB_RANK_1 | BB_RANK_8)) != 0 {
                for &p in &[QUEEN, ROOK, BISHOP, KNIGHT] {
                    dest.push(Move { from, to: to as i32, moving: makepiece(side, PAWN), captured: 0, promoted: makepiece(side, p), mask: MASK_PROMOTION });
                }
            } else {
                dest.push(Move { from, to: to as i32, moving: makepiece(side, PAWN), captured: 0, promoted: 0, mask: if offset.abs() == 16 { MASK_PAWN_2 } else { 0 } });
            }
        }
    }

    fn add_p_caps_to_vec(&self, side: usize, offset: i32, mut dests: u64, dest: &mut Vec<Move>) {
        let opp = 1 - side;
        while dests != 0 {
            let to = dests.trailing_zeros() as usize;
            dests &= dests - 1;
            let from = (to as i32) - offset;
            let captured = self.piece_at_sq(opp, to);
            let to_bit = 1u64 << to;
            if (to_bit & (BB_RANK_1 | BB_RANK_8)) != 0 {
                for &p in &[QUEEN, ROOK, BISHOP, KNIGHT] {
                    dest.push(Move { from, to: to as i32, moving: makepiece(side, PAWN), captured, promoted: makepiece(side, p), mask: MASK_PROMOTION | MASK_CAPTURE });
                }
            } else {
                dest.push(Move { from, to: to as i32, moving: makepiece(side, PAWN), captured, promoted: 0, mask: MASK_CAPTURE });
            }
        }
    }

    pub fn make_move(&mut self, m: Move) -> bool {
        let side = self.stm;
        let opp = 1 - side;
        let from_bit = 1u64 << m.from;
        let to_bit = 1u64 << m.to;

        self.history.push(HistoryEntry {
            from: m.from, to: m.to, captured: m.captured, promoted: m.promoted,
            c50: self.counter50, cr: self.castling_rights, ep: self.ep_square,
        });

        if ptype(m.moving) == PAWN || (m.mask & MASK_CAPTURE) != 0 { self.counter50 = 0; }
        else { self.counter50 += 1; }
        self.move_half_number += 1;
        self.move_number = self.move_half_number / 2;

        let p_type = ptype(m.moving) as usize;
        self.pieces[side][0] &= !from_bit;
        self.pieces[side][p_type] &= !from_bit;

        if (m.mask & MASK_CAPTURE) != 0 {
            let mut cap_sq = m.to as usize;
            if (m.mask & MASK_EP) != 0 {
                let sign = if side == WHITE { 1 } else { -1 };
                cap_sq = (m.to - 8 * sign) as usize;
            }
            let cap_bit = 1u64 << cap_sq;
            let cap_type = ptype(m.captured) as usize;
            self.pieces[opp][0] &= !cap_bit;
            self.pieces[opp][cap_type] &= !cap_bit;
        }

        let mut final_p_type = p_type;
        if (m.mask & MASK_PROMOTION) != 0 { final_p_type = ptype(m.promoted) as usize; }
        self.pieces[side][0] |= to_bit;
        self.pieces[side][final_p_type] |= to_bit;

        if (m.mask & MASK_CASTLING) != 0 {
            if side == WHITE {
                if m.to == 6 {
                    self.pieces[side][0] &= !(1u64 << 7); self.pieces[side][ROOK as usize] &= !(1u64 << 7);
                    self.pieces[side][0] |= 1u64 << 5; self.pieces[side][ROOK as usize] |= 1u64 << 5;
                } else {
                    self.pieces[side][0] &= !(1u64 << 0); self.pieces[side][ROOK as usize] &= !(1u64 << 0);
                    self.pieces[side][0] |= 1u64 << 3; self.pieces[side][ROOK as usize] |= 1u64 << 3;
                }
            } else {
                if m.to == 62 {
                    self.pieces[side][0] &= !(1u64 << 63); self.pieces[side][ROOK as usize] &= !(1u64 << 63);
                    self.pieces[side][0] |= 1u64 << 61; self.pieces[side][ROOK as usize] |= 1u64 << 61;
                } else {
                    self.pieces[side][0] &= !(1u64 << 56); self.pieces[side][ROOK as usize] &= !(1u64 << 56);
                    self.pieces[side][0] |= 1u64 << 59; self.pieces[side][ROOK as usize] |= 1u64 << 59;
                }
            }
        }

        self.ep_square = -1;
        if (m.mask & MASK_PAWN_2) != 0 { self.ep_square = (m.from + m.to) / 2; }

        if from_bit == (1u64 << 0) || to_bit == (1u64 << 0) { self.castling_rights &= !WHITE_CASTLE_Q; }
        if from_bit == (1u64 << 7) || to_bit == (1u64 << 7) { self.castling_rights &= !WHITE_CASTLE_K; }
        if from_bit == (1u64 << 56) || to_bit == (1u64 << 56) { self.castling_rights &= !BLACK_CASTLE_Q; }
        if from_bit == (1u64 << 63) || to_bit == (1u64 << 63) { self.castling_rights &= !BLACK_CASTLE_K; }
        if p_type == KING as usize {
            if side == WHITE { self.castling_rights &= !(WHITE_CASTLE_K | WHITE_CASTLE_Q); }
            else { self.castling_rights &= !(BLACK_CASTLE_K | BLACK_CASTLE_Q); }
        }

        self.update_occ();
        self.stm = 1 - self.stm;

        let king_sq = self.pieces[side][KING as usize].trailing_zeros() as usize;
        if king_sq < 64 && self.is_attacked(king_sq, opp) {
            self.undo_move();
            return false;
        }
        true
    }

    pub fn undo_move(&mut self) -> bool {
        if self.history.is_empty() { return false; }
        let h = self.history.pop().unwrap();
        self.stm = 1 - self.stm;
        let side = self.stm;
        let opp = 1 - side;
        
        let from_bit = 1u64 << h.from;
        let to_bit = 1u64 << h.to;
        
        self.counter50 = h.c50; self.castling_rights = h.cr; self.ep_square = h.ep;
        self.move_half_number -= 1; self.move_number = self.move_half_number / 2;

        let mut p_at_to = 0;
        for pt in 1..=6 { if (self.pieces[side][pt] & to_bit) != 0 { p_at_to = pt; break; } }
        self.pieces[side][0] &= !to_bit; self.pieces[side][p_at_to] &= !to_bit;
        
        let mut m_type = p_at_to;
        if h.promoted != 0 { m_type = PAWN as usize; }
        self.pieces[side][0] |= from_bit; self.pieces[side][m_type] |= from_bit;
        
        if h.captured != EMPTY {
            let mut cap_sq = h.to as usize;
            if ptype(makepiece(side, m_type as u8)) == PAWN && cap_sq == h.ep as usize {
                let sign = if side == WHITE { 1 } else { -1 };
                cap_sq = (h.to - 8 * sign) as usize;
            }
            let cap_bit = 1u64 << cap_sq;
            let cap_type = ptype(h.captured) as usize;
            self.pieces[opp][0] |= cap_bit; self.pieces[opp][cap_type] |= cap_bit;
        }

        if m_type == KING as usize {
            if side == WHITE {
                if h.from == 4 {
                    if h.to == 6 {
                        self.pieces[side][0] &= !(1u64 << 5); self.pieces[side][ROOK as usize] &= !(1u64 << 5);
                        self.pieces[side][0] |= 1u64 << 7; self.pieces[side][ROOK as usize] |= 1u64 << 7;
                    } else if h.to == 2 {
                        self.pieces[side][0] &= !(1u64 << 3); self.pieces[side][ROOK as usize] &= !(1u64 << 3);
                        self.pieces[side][0] |= 1u64 << 0; self.pieces[side][ROOK as usize] |= 1u64 << 0;
                    }
                }
            } else if h.from == 60 {
                if h.to == 62 {
                    self.pieces[side][0] &= !(1u64 << 61); self.pieces[side][ROOK as usize] &= !(1u64 << 61);
                    self.pieces[side][0] |= 1u64 << 63; self.pieces[side][ROOK as usize] |= 1u64 << 63;
                } else if h.to == 58 {
                    self.pieces[side][0] &= !(1u64 << 59); self.pieces[side][ROOK as usize] &= !(1u64 << 59);
                    self.pieces[side][0] |= 1u64 << 56; self.pieces[side][ROOK as usize] |= 1u64 << 56;
                }
            }
        }
        self.update_occ();
        true
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 { return 1; }
        let (moves, count) = {
           let mv = self.generate_moves_internal();
           let c = mv.len();
           (mv, c)
        };
        if depth == 1 { return count as u64; }
        let mut nodes = 0;
        for m in moves {
            if self.make_move(m) {
                nodes += self.perft(depth - 1);
                self.undo_move();
            }
        }
        nodes
    }

    pub fn is_attacked(&self, sq: usize, by_side: usize) -> bool {
        let occ = self.occ[2];
        if (p_atk_to(by_side, 1u64 << sq) & self.pieces[by_side][PAWN as usize]) != 0 { return true; }
        if (knight_attacks(sq) & self.pieces[by_side][KNIGHT as usize]) != 0 { return true; }
        if (king_attacks(sq) & self.pieces[by_side][KING as usize]) != 0 { return true; }
        if (bishop_attacks(sq, occ) & (self.pieces[by_side][BISHOP as usize] | self.pieces[by_side][QUEEN as usize])) != 0 { return true; }
        if (rook_attacks(sq, occ) & (self.pieces[by_side][ROOK as usize] | self.pieces[by_side][QUEEN as usize])) != 0 { return true; }
        false
    }
    fn piece_at_sq(&self, color: usize, sq: usize) -> u8 {
        let bit = 1u64 << sq;
        for pt in 1..=6 { if (self.pieces[color][pt] & bit) != 0 { return makepiece(color, pt as u8); } }
        0
    }
}

#[inline(always)]
fn p_atk_to(color: usize, bb: u64) -> u64 {
    if color == WHITE { ((bb >> 7) & NOT_A_FILE) | ((bb >> 9) & NOT_H_FILE) }
    else { ((bb << 7) & NOT_H_FILE) | ((bb << 9) & NOT_A_FILE) }
}
pub fn king_attacks(sq: usize) -> u64 {
    let bb = 1u64 << sq;
    let mut a = ((bb << 1) & NOT_A_FILE) | ((bb >> 1) & NOT_H_FILE);
    let b2 = bb | a;
    a |= (b2 << 8) | (b2 >> 8);
    a
}
pub fn knight_attacks(sq: usize) -> u64 {
    let b = 1u64 << sq;
    let mut a = ((b << 17) & NOT_A_FILE) | ((b << 15) & NOT_H_FILE);
    a |= ((b << 10) & NOT_AB_FILE) | ((b << 6) & NOT_GH_FILE);
    a |= ((b >> 6) & NOT_AB_FILE) | ((b >> 10) & NOT_GH_FILE);
    a |= ((b >> 15) & NOT_A_FILE) | ((b >> 17) & NOT_H_FILE);
    a
}
#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let index = (((occ | ROOK_MASK[sq]).wrapping_mul(ROOK_MAGIC[sq])) >> 52) as usize;
    LOOKUP_TABLE[ROOK_OFFSET[sq] + index]
}
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let index = (((occ | BISHOP_MASK[sq]).wrapping_mul(BISHOP_MAGIC[sq])) >> 55) as usize;
    LOOKUP_TABLE[BISHOP_OFFSET[sq] + index]
}
