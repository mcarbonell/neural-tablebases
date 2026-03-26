use crate::types::*;

pub struct X88Board {
    pub stm: usize,
    pub counter50: u32,
    pub castling_rights: u8,
    pub ep_square: i32,
    pub move_number: u32,
    pub move_half_number: u32,

    pub piece_at: [u8; 128],
    pub king_squares: [i32; 2],

    // Tactical info for GNN
    pub attacks: [[u8; 128]; 2],      // Bitfield of piece types attacking each square
    pub num_attacks: [[u8; 128]; 2],  // Number of pieces attacking each square
    pub pin_direction: [[i32; 128]; 2], // Direction of absolute pins (0 = not pinned)
    pub in_check_valid_squares: [[u8; 128]; 2], // Valid squares when in check (blocking or capturing)
    
    pub history: Vec<HistoryEntry>,
}

impl X88Board {
    pub fn new() -> X88Board {
        X88Board {
            stm: WHITE,
            counter50: 0,
            castling_rights: 0,
            ep_square: -1,
            move_number: 1,
            move_half_number: 1,
            piece_at: [EMPTY; 128],
            king_squares: [-1; 2],
            attacks: [[0; 128]; 2],
            num_attacks: [[0; 128]; 2],
            pin_direction: [[0; 128]; 2],
            in_check_valid_squares: [[0; 128]; 2],
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
        self.piece_at = [EMPTY; 128];
        self.king_squares = [-1; 2];
        self.history.clear();
        self.clear_tactical();
    }

    fn clear_tactical(&mut self) {
        self.attacks = [[0; 128]; 2];
        self.num_attacks = [[0; 128]; 2];
        self.pin_direction = [[0; 128]; 2];
        self.in_check_valid_squares = [[0; 128]; 2];
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
                let sq = square(f, r);
                self.piece_at[sq as usize] = piece;
                if ptype(piece) == KING { self.king_squares[pcolor(piece) as usize] = sq; }
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
        
        self.generate_tactical_info();
    }

    pub fn generate_tactical_info(&mut self) {
        self.clear_tactical();
        
        for who in 0..2 {
            // Kings
            let k_sq = self.king_squares[who];
            if k_sq != -1 {
                for &off in &OFFSETS_QUEEN_KING {
                    let to = k_sq + off;
                    if valid_square(to) { self.add_attack(who, KING, k_sq, to); }
                }
            }

            // Knights
            for sq in 0..128 {
                if !valid_square(sq as i32) { continue; }
                let p = self.piece_at[sq];
                if p != EMPTY && pcolor(p) as usize == who && ptype(p) == KNIGHT {
                    for &off in &OFFSETS_KNIGHT {
                        let to = sq as i32 + off;
                        if valid_square(to) { self.add_attack(who, KNIGHT, sq as i32, to); }
                    }
                }
            }

            // Pawns
            for sq in 0..128 {
                if !valid_square(sq as i32) { continue; }
                let p = self.piece_at[sq];
                if p != EMPTY && pcolor(p) as usize == who && ptype(p) == PAWN {
                    let sign = if who == WHITE { 1 } else { -1 };
                    for &off in &[(16 * sign - 1), (16 * sign + 1)] {
                        let to = sq as i32 + off;
                        if valid_square(to) { self.add_attack(who, PAWN, sq as i32, to); }
                    }
                }
            }

            // Sliders
            for sq in 0..128 {
                if !valid_square(sq as i32) { continue; }
                let p = self.piece_at[sq];
                if p == EMPTY || pcolor(p) as usize != who { continue; }
                let p_type = ptype(p);
                if p_type == BISHOP || p_type == ROOK || p_type == QUEEN {
                    let offsets: &[i32] = match p_type {
                        BISHOP => &OFFSETS_BISHOP,
                        ROOK => &OFFSETS_ROOK,
                        _ => &OFFSETS_QUEEN_KING,
                    };
                    for &off in offsets {
                        let mut to = sq as i32 + off;
                        while valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY {
                                self.add_attack(who, p_type, sq as i32, to);
                            } else {
                                // Hits a piece - SliderAttack logic
                                self.add_slider_attack(who, p_type, sq as i32, to, off);
                                break;
                            }
                            to += off;
                        }
                    }
                }
            }
        }
    }

    fn add_attack(&mut self, who: usize, p_type: u8, _from: i32, to: i32) {
        self.attacks[who][to as usize] |= attackbit(p_type);
        self.num_attacks[who][to as usize] += 1;
        
        let opp = opposite(who);
        if to == self.king_squares[opp] {
            // Check! (Could implement check evasion squares here but usually needed for movegen)
        }
    }

    fn add_slider_attack(&mut self, who: usize, p_type: u8, from: i32, to: i32, direction: i32) {
        self.attacks[who][to as usize] |= attackbit(p_type);
        self.num_attacks[who][to as usize] += 1;

        let opp = opposite(who);
        let attacked_p = self.piece_at[to as usize];
        
        // Pin detection
        if to == self.king_squares[opp] {
            // Ray hits king - this is check
            // Fill in_check_valid_squares
            let mut sq = ite_prev(to, from, direction);
            while sq != from {
                self.in_check_valid_squares[opp][sq as usize] = 1;
                sq = ite_prev(sq, from, direction);
            }
            self.in_check_valid_squares[opp][from as usize] = 1;
        } else if pcolor(attacked_p) as usize == opp {
            // Hits enemy piece - check if pinned
            let mut next = to + direction;
            while valid_square(next) {
                let next_p = self.piece_at[next as usize];
                if next_p != EMPTY {
                    if next == self.king_squares[opp] {
                        // PIN!
                        self.pin_direction[opp][to as usize] = direction;
                    }
                    break;
                }
                next += direction;
            }
        }
    }

    pub fn fill_gnn_features(&mut self, out: &mut crate::GnnFeatureStruct) {
        // Reset output
        out.piece_ids = [0; 64];
        out.white_atk_counts = [0; 64];
        out.black_atk_counts = [0; 64];
        out.attacker_bitmasks = [0; 64];
        out.flags = [0; 64];
        out.edge_count = 0;

        for r in 0..8 {
            for f in 0..8 {
                let sq_x88 = square(f, r) as usize;
                let sq_64 = (r * 8 + f) as usize;
                
                let p = self.piece_at[sq_x88];
                let mut p_id = 0;
                if p != EMPTY {
                    let p_type = ptype(p);
                    let color = pcolor(p) as usize;
                    p_id = if color == WHITE { p_type as i8 } else { (p_type + 6) as i8 };
                }
                out.piece_ids[sq_64] = p_id;
                
                out.white_atk_counts[sq_64] = self.num_attacks[WHITE][sq_x88] as i8;
                out.black_atk_counts[sq_64] = self.num_attacks[BLACK][sq_x88] as i8;
                out.attacker_bitmasks[sq_64] = self.attacks[WHITE][sq_x88] | self.attacks[BLACK][sq_x88];

                // Flags: 
                // bit 0: King Square under check
                // bit 1: Pinned square (absolute pin)
                // bit 2: Safe square (not attacked by enemy)
                // bit 3: Hanging piece (attacked and not defended)
                // bit 4: Protected square (defended by us)
                let mut flag = 0u8;
                let who = if p != EMPTY { pcolor(p) as usize } else { self.stm };
                let opp = opposite(who);
                
                if sq_x88 as i32 == self.king_squares[self.stm] && self.is_attacked(sq_x88 as i32, opposite(self.stm)) {
                    flag |= 1;
                }
                if self.pin_direction[WHITE][sq_x88] != 0 || self.pin_direction[BLACK][sq_x88] != 0 {
                    flag |= 2;
                }
                if self.num_attacks[opp][sq_x88] == 0 {
                    flag |= 4; // Safe
                }
                if p != EMPTY && self.num_attacks[opp][sq_x88] > 0 && self.num_attacks[who][sq_x88] == 0 {
                    flag |= 8; // Hanging
                }
                if self.num_attacks[who][sq_x88] > 0 {
                    flag |= 16; // Protected
                }
                out.flags[sq_64] = flag;
            }
        }

        // Mobility edges (Type 0)
        let moves = self.generate_legal_moves();
        for m in moves {
            if out.edge_count >= 1024 { break; }
            let from64 = (rank(m.from) * 8 + file(m.from)) as u16;
            let to64 = (rank(m.to) * 8 + file(m.to)) as u16;
            
            // Edge type bitmask in high 4 bits
            let mut t = 0u16;
            if (m.mask & MASK_CAPTURE) != 0 { t |= 1; }
            if (m.mask & MASK_CHECK) != 0 { t |= 2; }
            if (m.mask & MASK_PROMOTION) != 0 { t |= 4; }
            if (m.mask & (MASK_CASTLING | MASK_EP | MASK_PAWN_2)) != 0 { t |= 8; }
            
            let edge = (t << 12) | (from64 << 6) | to64;
            out.edges[out.edge_count as usize] = edge;
            out.edge_count += 1;
        }
    }

    fn is_attacked(&self, sq: i32, by_whom: usize) -> bool {
        self.num_attacks[by_whom][sq as usize] > 0
    }

    pub fn generate_legal_moves(&mut self) -> Vec<Move> {
        let mut pseudo = Vec::with_capacity(128);
        let side = self.stm;
        let opp = opposite(side);
        
        for sq in 0..128 {
            if !valid_square(sq as i32) { continue; }
            let p = self.piece_at[sq];
            if p == EMPTY || pcolor(p) as usize != side { continue; }
            
            let p_type = ptype(p);
            match p_type {
                PAWN => {
                    let sign = if side == WHITE { 1 } else { -1 };
                    let to1 = sq as i32 + 16 * sign;
                    if valid_square(to1) && self.piece_at[to1 as usize] == EMPTY {
                        pseudo.push(Move { from: sq as i32, to: to1, moving: p, captured: 0, promoted: 0, mask: 0 });
                        let to2 = to1 + 16 * sign;
                        if rank(sq as i32) == (if side == WHITE { RANK_2 } else { RANK_7 }) && self.piece_at[to2 as usize] == EMPTY {
                            pseudo.push(Move { from: sq as i32, to: to2, moving: p, captured: 0, promoted: 0, mask: MASK_PAWN_2 });
                        }
                    }
                    for &off in &[(16 * sign - 1), (16 * sign + 1)] {
                        let to = sq as i32 + off;
                        if !valid_square(to) { continue; }
                        let target = self.piece_at[to as usize];
                        if target != EMPTY && pcolor(target) as usize == opp {
                            pseudo.push(Move { from: sq as i32, to, moving: p, captured: target, promoted: 0, mask: MASK_CAPTURE });
                        } else if to == self.ep_square {
                            pseudo.push(Move { from: sq as i32, to, moving: p, captured: makepiece(opp, PAWN), promoted: 0, mask: MASK_EP | MASK_CAPTURE });
                        }
                    }
                }
                KNIGHT | KING => {
                    let offsets: &[i32] = if p_type == KNIGHT { &OFFSETS_KNIGHT } else { &OFFSETS_QUEEN_KING };
                    for &off in offsets {
                        let to = sq as i32 + off;
                        if valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY || pcolor(target) as usize == opp {
                                pseudo.push(Move { from: sq as i32, to, moving: p, captured: target, promoted: 0, mask: if target != EMPTY { MASK_CAPTURE } else { 0 } });
                            }
                        }
                    }
                }
                _ => { // Sliders
                    let offsets: &[i32] = match p_type {
                        BISHOP => &OFFSETS_BISHOP,
                        ROOK => &OFFSETS_ROOK,
                        _ => &OFFSETS_QUEEN_KING,
                    };
                    for &off in offsets {
                        let mut to = sq as i32 + off;
                        while valid_square(to) {
                            let target = self.piece_at[to as usize];
                            if target == EMPTY || pcolor(target) as usize == opp {
                                pseudo.push(Move { from: sq as i32, to, moving: p, captured: target, promoted: 0, mask: if target != EMPTY { MASK_CAPTURE } else { 0 } });
                            }
                            if target != EMPTY { break; }
                            to += off;
                        }
                    }
                }
            }
        }
        
        let mut legal = Vec::with_capacity(pseudo.len());
        for mut m in pseudo {
            if self.make_move(m) {
                // Check if this move gives check
                let opp_king = self.king_squares[self.stm];
                if self.is_attacked_by_simple(opp_king, opposite(self.stm)) {
                    m.mask |= MASK_CHECK;
                }
                legal.push(m);
                self.undo_move();
            }
        }
        
        // Castling
        if side == WHITE {
            if (self.castling_rights & WHITE_CASTLE_K) != 0 && self.piece_at[5] == EMPTY && self.piece_at[6] == EMPTY {
                if !self.is_attacked_by_simple(4, BLACK) && !self.is_attacked_by_simple(5, BLACK) && !self.is_attacked_by_simple(6, BLACK) {
                    legal.push(Move { from: 4, to: 6, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & WHITE_CASTLE_Q) != 0 && self.piece_at[1] == EMPTY && self.piece_at[2] == EMPTY && self.piece_at[3] == EMPTY {
                if !self.is_attacked_by_simple(4, BLACK) && !self.is_attacked_by_simple(3, BLACK) && !self.is_attacked_by_simple(2, BLACK) {
                    legal.push(Move { from: 4, to: 2, moving: WK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        } else {
            if (self.castling_rights & BLACK_CASTLE_K) != 0 && self.piece_at[117] == EMPTY && self.piece_at[118] == EMPTY {
                if !self.is_attacked_by_simple(116, WHITE) && !self.is_attacked_by_simple(117, WHITE) && !self.is_attacked_by_simple(118, WHITE) {
                    legal.push(Move { from: 116, to: 118, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
            if (self.castling_rights & BLACK_CASTLE_Q) != 0 && self.piece_at[113] == EMPTY && self.piece_at[114] == EMPTY  && self.piece_at[115] == EMPTY {
                if !self.is_attacked_by_simple(116, WHITE) && !self.is_attacked_by_simple(115, WHITE) && !self.is_attacked_by_simple(114, WHITE) {
                    legal.push(Move { from: 116, to: 114, moving: BK, captured: 0, promoted: 0, mask: MASK_CASTLING });
                }
            }
        }

        legal
    }

    pub fn make_move(&mut self, m: Move) -> bool {
        let side = self.stm;
        let opp = opposite(side);
        
        self.history.push(HistoryEntry {
            from: m.from, to: m.to, captured: m.captured, promoted: m.promoted,
            c50: self.counter50, cr: self.castling_rights, ep: self.ep_square,
        });
        
        // Update 50-move rule
        if ptype(m.moving) == PAWN || (m.mask & MASK_CAPTURE) != 0 { self.counter50 = 0; }
        else { self.counter50 += 1; }

        self.piece_at[m.from as usize] = EMPTY;
        self.piece_at[m.to as usize] = if m.promoted != 0 { m.promoted } else { m.moving };
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

        // Update rights
        if m.from == 0 || m.to == 0 { self.castling_rights &= !WHITE_CASTLE_Q; }
        if m.from == 7 || m.to == 7 { self.castling_rights &= !WHITE_CASTLE_K; }
        if m.from == 112 || m.to == 112 { self.castling_rights &= !BLACK_CASTLE_Q; }
        if m.from == 119 || m.to == 119 { self.castling_rights &= !BLACK_CASTLE_K; }
        if ptype(m.moving) == KING {
            if side == WHITE { self.castling_rights &= !(WHITE_CASTLE_K | WHITE_CASTLE_Q); }
            else { self.castling_rights &= !(BLACK_CASTLE_K | BLACK_CASTLE_Q); }
        }
        
        self.stm = opp;
        self.ep_square = if (m.mask & MASK_PAWN_2) != 0 { (m.from + m.to) / 2 } else { -1 };
        
        let king_sq = self.king_squares[side];
        if self.is_attacked_by_simple(king_sq, opp) {
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
        
        let p = self.piece_at[h.to as usize];
        self.piece_at[h.from as usize] = if h.promoted != 0 { makepiece(side, PAWN) } else { p };
        self.piece_at[h.to as usize] = EMPTY;

        if (h.from - h.to).abs() == 2 && ptype(self.piece_at[h.from as usize]) == KING {
            // Restore rooks on castling
            if side == WHITE {
                if h.to == 6 { self.piece_at[5] = EMPTY; self.piece_at[7] = WR; }
                else { self.piece_at[3] = EMPTY; self.piece_at[0] = WR; }
            } else {
                if h.to == 118 { self.piece_at[117] = EMPTY; self.piece_at[119] = BR; }
                else { self.piece_at[115] = EMPTY; self.piece_at[112] = BR; }
            }
        }

        if h.captured != EMPTY {
            let mut cap_sq = h.to;
            if ptype(self.piece_at[h.from as usize]) == PAWN && h.to == h.ep {
                let sign = if side == WHITE { 1 } else { -1 };
                cap_sq = h.to - 16 * sign;
            }
            self.piece_at[cap_sq as usize] = h.captured;
        }
        if ptype(self.piece_at[h.from as usize]) == KING { self.king_squares[side] = h.from; }
        self.ep_square = h.ep;
        self.castling_rights = h.cr;
        self.counter50 = h.c50;
        true
    }

    fn is_attacked_by_simple(&self, sq: i32, by_whom: usize) -> bool {
        // Simple scan for legal move verification
        let opp = opposite(by_whom);
        let sign = if by_whom == WHITE { 1 } else { -1 };
        for &off in &[(16 * sign - 1), (16 * sign + 1)] {
            let from = sq - off;
            if valid_square(from) && self.piece_at[from as usize] == makepiece(by_whom, PAWN) { return true; }
        }
        for &off in &OFFSETS_KNIGHT {
            let from = sq + off;
            if valid_square(from) && self.piece_at[from as usize] == makepiece(by_whom, KNIGHT) { return true; }
        }
        for &off in &OFFSETS_QUEEN_KING {
            let from = sq + off;
            if valid_square(from) && self.piece_at[from as usize] == makepiece(by_whom, KING) { return true; }
        }
        for &p_type in &[BISHOP, ROOK, QUEEN] {
            let offsets: &[i32] = match p_type {
                BISHOP => &OFFSETS_BISHOP,
                ROOK => &OFFSETS_ROOK,
                _ => &OFFSETS_QUEEN_KING,
            };
            for &off in offsets {
                let mut from = sq + off;
                while valid_square(from) {
                    let p = self.piece_at[from as usize];
                    if p != EMPTY {
                        if pcolor(p) as usize == by_whom && (ptype(p) == p_type || ptype(p) == QUEEN) { return true; }
                        break;
                    }
                    from += off;
                }
            }
        }
        false
    }
}

fn ite_prev(current: i32, target: i32, direction: i32) -> i32 {
    if current == target { return current; }
    current - direction
}
