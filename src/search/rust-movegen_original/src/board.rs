use crate::types::*;

pub struct Board {
    pub stm: usize,
    pub counter50: u32,
    pub castling_rights: u8,
    pub ep_square: i32,
    pub move_number: u32,
    pub move_half_number: u32,

    pub piece_at: [u8; 128],
    pub king_squares: [i32; 2],

    // Move list storage (ply-indexed)
    pub moves: [Vec<Move>; 256],
    pub move_count: [u32; 256],
    pub history_ply: usize,
    pub history: [HistoryEntry; 1024],
}

impl Board {
    pub fn new() -> Board {
        let mut b = Board {
            stm: WHITE,
            counter50: 0,
            castling_rights: 0,
            ep_square: -1,
            move_number: 1,
            move_half_number: 1,
            piece_at: [EMPTY; 128],
            king_squares: [-1; 2],
            moves: std::array::from_fn(|_| Vec::with_capacity(256)),
            move_count: [0; 256],
            history_ply: 0,
            history: [HistoryEntry::default(); 1024],
        };
        b.reset();
        b
    }

    pub fn reset(&mut self) {
        self.stm = WHITE;
        self.counter50 = 0;
        self.castling_rights = 0;
        self.ep_square = -1;
        self.move_number = 1;
        self.piece_at = [EMPTY; 128];
        self.king_squares = [-1; 2];
        self.history_ply = 0;
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
                self.add_piece(pcolor(piece) as usize, ptype(piece), square(f, r));
                f += 1;
            }
        }

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
            let sf = (parts[3].chars().nth(0).unwrap() as i32) - 97;
            let sr = (parts[3].chars().nth(1).unwrap() as i32) - 49;
            self.ep_square = square(sf, sr);
        }
        if parts.len() > 4 { self.counter50 = parts[4].parse().unwrap_or(0); }
        if parts.len() > 5 { 
            self.move_number = parts[5].parse().unwrap_or(1);
            self.move_half_number = self.move_number * 2 - (if self.stm == WHITE { 1 } else { 0 });
        }
    }

    pub fn add_piece(&mut self, color: usize, p_type: u8, sq: i32) {
        let piece = makepiece(color, p_type);
        self.piece_at[sq as usize] = piece;
        if p_type == KING { self.king_squares[color] = sq; }
    }

    pub fn generate_moves(&mut self) -> &Vec<Move> {
        let ply = self.history_ply;
        self.moves[ply].clear();
        let mut pseudo_moves = Vec::with_capacity(128);
        
        let side = self.stm;
        let opp = 1 - side;
        
        for sq in 0..128 {
            if !valid_square(sq) { continue; }
            let piece = self.piece_at[sq as usize];
            if piece == EMPTY || pcolor(piece) as usize != side { continue; }
            
            let p_type = ptype(piece);
            match p_type {
                PAWN => {
                    let sign = if side == WHITE { 1 } else { -1 };
                    // Push
                    let to1 = sq + 16 * sign;
                    if valid_square(to1) && self.piece_at[to1 as usize] == EMPTY {
                        self.add_pawn_move_to(sq, to1, &mut pseudo_moves);
                        let to2 = to1 + 16 * sign;
                        let start_rank = if side == WHITE { RANK_2 } else { RANK_7 };
                        if rank(sq) == start_rank && valid_square(to2) && self.piece_at[to2 as usize] == EMPTY {
                            pseudo_moves.push(Move { from: sq, to: to2, moving: piece, captured: 0, promoted: 0, mask: MASK_PAWN_2 });
                        }
                    }
                    // Capture
                    for &off in &[(16 * sign - 1), (16 * sign + 1)] {
                        let to = sq + off;
                        if !valid_square(to) { continue; }
                        let target = self.piece_at[to as usize];
                        if target != EMPTY && pcolor(target) as usize == opp {
                            self.add_pawn_move_to(sq, to, &mut pseudo_moves);
                        } else if to == self.ep_square {
                            pseudo_moves.push(Move { from: sq, to, moving: piece, captured: makepiece(opp, PAWN), promoted: 0, mask: MASK_EP | MASK_CAPTURE });
                        }
                    }
                }
                KNIGHT => {
                    for &off in &OFFSETS_KNIGHT {
                        let to = sq + off;
                        if valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY || pcolor(target) as usize == opp {
                                pseudo_moves.push(Move { from: sq, to, moving: piece, captured: target, promoted: 0, mask: if target != EMPTY { MASK_CAPTURE } else { 0 } });
                            }
                        }
                    }
                }
                KING => {
                    for &off in &OFFSETS_QUEEN_KING {
                        let to = sq + off;
                        if valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY || pcolor(target) as usize == opp {
                                pseudo_moves.push(Move { from: sq, to, moving: piece, captured: target, promoted: 0, mask: if target != EMPTY { MASK_CAPTURE } else { 0 } });
                            }
                        }
                    }
                }
                _ => { // Sliders
                    let offsets: &[i32] = if p_type == BISHOP { &OFFSETS_BISHOP } else if p_type == ROOK { &OFFSETS_ROOK } else { &OFFSETS_QUEEN_KING };
                    for &off in offsets {
                        let mut to = sq + off;
                        while valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY || pcolor(target) as usize == opp {
                                pseudo_moves.push(Move { from: sq, to, moving: piece, captured: target, promoted: 0, mask: if target != EMPTY { MASK_CAPTURE } else { 0 } });
                            }
                            if target != EMPTY { break; }
                            to += off;
                        }
                    }
                }
            }
        }

        // Castle
        if side == WHITE {
            if (self.castling_rights & WHITE_CASTLE_K) != 0 && self.piece_at[5] == EMPTY && self.piece_at[6] == EMPTY {
                if !self.is_attacked(4, BLACK) && !self.is_attacked(5, BLACK) && !self.is_attacked(6, BLACK) {
                    pseudo_moves.push(Move { from: 4, to: 6, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & WHITE_CASTLE_Q) != 0 && self.piece_at[1] == EMPTY && self.piece_at[2] == EMPTY && self.piece_at[3] == EMPTY {
                if !self.is_attacked(4, BLACK) && !self.is_attacked(3, BLACK) && !self.is_attacked(2, BLACK) {
                    pseudo_moves.push(Move { from: 4, to: 2, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        } else {
            if (self.castling_rights & BLACK_CASTLE_K) != 0 && self.piece_at[117] == EMPTY && self.piece_at[118] == EMPTY {
                if !self.is_attacked(116, WHITE) && !self.is_attacked(117, WHITE) && !self.is_attacked(118, WHITE) {
                    pseudo_moves.push(Move { from: 116, to: 118, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & BLACK_CASTLE_Q) != 0 && self.piece_at[113] == EMPTY && self.piece_at[114] == EMPTY && self.piece_at[115] == EMPTY {
                if !self.is_attacked(116, WHITE) && !self.is_attacked(115, WHITE) && !self.is_attacked(114, WHITE) {
                    pseudo_moves.push(Move { from: 116, to: 114, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        }

        for m in pseudo_moves {
            if self.make_move(m) {
                self.moves[ply].push(m);
                self.undo_move();
            }
        }
        self.move_count[ply] = self.moves[ply].len() as u32;
        &self.moves[ply]
    }

    fn add_pawn_move_to(&self, from: i32, to: i32, dest: &mut Vec<Move>) {
        let piece = self.piece_at[from as usize];
        let side = pcolor(piece) as usize;
        let target = self.piece_at[to as usize];
        let mask = if target != EMPTY { MASK_CAPTURE } else { 0 };
        if rank(to) == RANK_8 || rank(to) == RANK_1 {
            for &p in &[QUEEN, ROOK, BISHOP, KNIGHT] {
                dest.push(Move { from, to, moving: piece, captured: target, promoted: makepiece(side, p), mask: mask | MASK_PROMOTION });
            }
        } else {
            dest.push(Move { from, to, moving: piece, captured: target, promoted: 0, mask });
        }
    }

    pub fn make_move(&mut self, m: Move) -> bool {
        self.history[self.history_ply] = HistoryEntry {
            from: m.from, to: m.to, captured: m.captured, promoted: m.promoted,
            c50: self.counter50, cr: self.castling_rights, ep: self.ep_square,
        };
        self.history_ply += 1;

        let side = self.stm;
        let opp = 1 - side;
        
        if ptype(m.moving) == PAWN || (m.mask & MASK_CAPTURE) != 0 { self.counter50 = 0; }
        else { self.counter50 += 1; }
        self.move_half_number += 1;
        self.move_number = self.move_half_number / 2;

        self.piece_at[m.from as usize] = EMPTY;
        let mut final_piece = m.moving;
        if (m.mask & MASK_PROMOTION) != 0 { final_piece = m.promoted; }
        self.piece_at[m.to as usize] = final_piece;
        
        if ptype(m.moving) == KING { self.king_squares[side] = m.to; }

        if (m.mask & MASK_EP) != 0 {
            let sign = if side == WHITE { 1 } else { -1 };
            self.piece_at[(m.to - 16 * sign) as usize] = EMPTY;
        }

        if (m.mask & MASK_CASTLING) != 0 {
            if side == WHITE {
                if m.to == 6 { self.piece_at[7] = EMPTY; self.piece_at[5] = WR; }
                else { self.piece_at[0] = EMPTY; self.piece_at[3] = WR; }
            } else {
                if m.to == 118 { self.piece_at[119] = EMPTY; self.piece_at[117] = BR; }
                else { self.piece_at[112] = EMPTY; self.piece_at[115] = BR; }
            }
        }

        self.ep_square = -1;
        if (m.mask & MASK_PAWN_2) != 0 { self.ep_square = (m.from + m.to) / 2; }

        if m.from == 0 || m.to == 0 { self.castling_rights &= !WHITE_CASTLE_Q; }
        if m.from == 7 || m.to == 7 { self.castling_rights &= !WHITE_CASTLE_K; }
        if m.from == 112 || m.to == 112 { self.castling_rights &= !BLACK_CASTLE_Q; }
        if m.from == 119 || m.to == 119 { self.castling_rights &= !BLACK_CASTLE_K; }
        if ptype(m.moving) == KING {
            if side == WHITE { self.castling_rights &= !(WHITE_CASTLE_K | WHITE_CASTLE_Q); }
            else { self.castling_rights &= !(BLACK_CASTLE_K | BLACK_CASTLE_Q); }
        }

        self.stm = 1 - self.stm;
        let king_sq = self.king_squares[side];
        if self.is_attacked(king_sq, opp) {
            self.undo_move();
            return false;
        }
        true
    }

    pub fn undo_move(&mut self) -> bool {
        if self.history_ply == 0 { return false; }
        self.history_ply -= 1;
        let h = self.history[self.history_ply];
        self.stm = 1 - self.stm;
        let side = self.stm;
        let opp = 1 - side;

        let moving_piece = self.piece_at[h.to as usize];
        self.piece_at[h.to as usize] = EMPTY;
        let mut original_moving = moving_piece;
        if h.promoted != 0 { original_moving = makepiece(side, PAWN); }
        self.piece_at[h.from as usize] = original_moving;
        
        if h.captured != EMPTY {
            let mut cap_sq = h.to;
            if ptype(original_moving) == PAWN && h.to == h.ep {
                let sign = if side == WHITE { 1 } else { -1 };
                cap_sq = h.to - 16 * sign;
            }
            self.piece_at[cap_sq as usize] = h.captured;
        }

        if ptype(original_moving) == KING {
            self.king_squares[side] = h.from;
            // Restore rooks
            if (h.to - h.from).abs() == 2 {
                if side == WHITE {
                    if h.to == 6 { self.piece_at[5] = EMPTY; self.piece_at[7] = WR; }
                    else { self.piece_at[3] = EMPTY; self.piece_at[0] = WR; }
                } else {
                    if h.to == 118 { self.piece_at[117] = EMPTY; self.piece_at[119] = BR; }
                    else { self.piece_at[115] = EMPTY; self.piece_at[112] = BR; }
                }
            }
        }

        self.counter50 = h.c50;
        self.castling_rights = h.cr;
        self.ep_square = h.ep;
        self.move_half_number -= 1;
        self.move_number = self.move_half_number / 2;
        true
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 { return 1; }
        let (moves, count) = {
            let moves_ref = self.generate_moves();
            (moves_ref.clone(), moves_ref.len())
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

    pub fn is_attacked(&self, sq: i32, by_side: usize) -> bool {
        let opp = 1 - by_side;
        // Pawn
        let sign = if by_side == WHITE { 1 } else { -1 };
        for &off in &[(16 * sign - 1), (16 * sign + 1)] {
            let from = sq - off;
            if valid_square(from) {
                let p = self.piece_at[from as usize];
                if p == makepiece(by_side, PAWN) { return true; }
            }
        }
        // Knight
        for &off in &OFFSETS_KNIGHT {
            let from = sq + off;
            if valid_square(from) {
                let p = self.piece_at[from as usize];
                if p == makepiece(by_side, KNIGHT) { return true; }
            }
        }
        // King
        for &off in &OFFSETS_QUEEN_KING {
            let from = sq + off;
            if valid_square(from) {
                let p = self.piece_at[from as usize];
                if p == makepiece(by_side, KING) { return true; }
            }
        }
        // Sliders
        for &p_type in &[BISHOP, ROOK, QUEEN] {
            let offsets: &[i32] = if p_type == BISHOP { &OFFSETS_BISHOP } else if p_type == ROOK { &OFFSETS_ROOK } else { &OFFSETS_QUEEN_KING };
            for &off in offsets {
                let mut from = sq + off;
                while valid_square(from) {
                    let p = self.piece_at[from as usize];
                    if p != EMPTY {
                        if pcolor(p) as usize == by_side && (ptype(p) == p_type || ptype(p) == QUEEN) { return true; }
                        break;
                    }
                    from += off;
                }
            }
        }
        false
    }
}
