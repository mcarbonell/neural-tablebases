"use strict";

// Piece type and color constants (needed for Node.js)
var white = 0;
var black = 1;
var empty = 0;
var p = 1;
var n = 2;
var b = 3;
var r = 4;
var q = 5;
var k = 6;
var wp = 1;
var wn = 2;
var wb = 3;
var wr = 4;
var wq = 5;
var wk = 6;
var bp = wp + 8;
var bn = wn + 8;
var bb = wb + 8;
var br = wr + 8;
var bq = wq + 8;
var bk = wk + 8;

// Maximum ply depth for history
var MAXPLY = 100;

// Helper functions for FEN parsing and move notation
function fileFromChar(char) {
    return char.charCodeAt(0) - 'a'.charCodeAt(0);
}

function rankFromChar(char) {
    return char.charCodeAt(0) - '1'.charCodeAt(0);
}

function fileToChar(file) {
    return String.fromCharCode('a'.charCodeAt(0) + file);
}

function rankToChar(rank) {
    return String.fromCharCode('1'.charCodeAt(0) + rank);
}

function pcolor(piece) {
    return (piece === 0) ? -1 : (piece & 8) >>> 3;
}

function ptype(piece) {
    return (piece & 7) >>> 0;
}

function makepiece(pcolor, ptype) {
    return (pcolor << 3) + ptype;
}

function pieceChar(piece) {
    return charpieces[piece];
}

function opposite(turn) {
    return 1 - turn;
}

// Piece character arrays
var pieces = '.♟♚♞♝♜♛  ♙♔♘♗♖♕';
var diagpieces = [' ', '♟', '♞', '♝', '♜', '♛', '♚', '', '', '♙', '♘', '♗', 'r', '♖', '♔', ''];
var charpieces = [' ', 'P', 'N', 'B', 'R', 'Q', 'K', '', '', 'p', 'n', 'b', 'r', 'q', 'k', ''];

// Castling rights
var whitecancastleK = 1;
var whitecancastleQ = 2;
var blackcancastleK = 4;
var blackcancastleQ = 8;

// Move mask constants
var mask_capture = 1;
var mask_promotion = 1 << 1;
var mask_pawnmove = 1 << 2;
var mask_pawn2 = 1 << 15;
var mask_ep = 1 << 3;
var mask_castling = 1 << 4;
var mask_check = 1 << 5;
var mask_doublecheck = 1 << 6;
var mask_discovercheck = 1 << 7;
var mask_safe = 1 << 8;
var mask_protected = 1 << 9;
var mask_insecure = 1 << 10;
var mask_hanging = 1 << 11;
var mask_goodcapture = 1 << 12;
var mask_freecapture = 1 << 13;
var mask_winningcapture = 1 << 14;

// Bit manipulation helpers
function lowestSetBit(n) {
    if (n === 0) return 32;
    let count = 0;
    while ((n & 1) === 0) {
        n >>>= 1;
        count++;
    }
    return count;
}

function bitCount(n) {
    let count = 0;
    while (n) {
        count += n & 1;
        n >>>= 1;
    }
    return count;
}




var a1 = 0;
var b1 = 1;
var c1 = 2;
var d1 = 3;
var e1 = 4;
var f1 = 5;
var g1 = 6;
var h1 = 7;

var a2 = a1 + 16;
var b2 = b1 + 16;
var c2 = c1 + 16;
var d2 = d1 + 16;
var e2 = e1 + 16;
var f2 = f1 + 16;
var g2 = g1 + 16;
var h2 = h1 + 16;

var a3 = a2 + 16;
var b3 = b2 + 16;
var c3 = c2 + 16;
var d3 = d2 + 16;
var e3 = e2 + 16;
var f3 = f2 + 16;
var g3 = g2 + 16;
var h3 = h2 + 16;

var a4 = a3 + 16;
var b4 = b3 + 16;
var c4 = c3 + 16;
var d4 = d3 + 16;
var e4 = e3 + 16;
var f4 = f3 + 16;
var g4 = g3 + 16;
var h4 = h3 + 16;

var a5 = a4 + 16;
var b5 = b4 + 16;
var c5 = c4 + 16;
var d5 = d4 + 16;
var e5 = e4 + 16;
var f5 = f4 + 16;
var g5 = g4 + 16;
var h5 = h4 + 16;

var a6 = a5 + 16;
var b6 = b5 + 16;
var c6 = c5 + 16;
var d6 = d5 + 16;
var e6 = e5 + 16;
var f6 = f5 + 16;
var g6 = g5 + 16;
var h6 = h5 + 16;

var a7 = a6 + 16;
var b7 = b6 + 16;
var c7 = c6 + 16;
var d7 = d6 + 16;
var e7 = e6 + 16;
var f7 = f6 + 16;
var g7 = g6 + 16;
var h7 = h6 + 16;

var a8 = a7 + 16;
var b8 = b7 + 16;
var c8 = c7 + 16;
var d8 = d7 + 16;
var e8 = e7 + 16;
var f8 = f7 + 16;
var g8 = g7 + 16;
var h8 = h7 + 16; // 119 

var filea = 0;
var fileb = 1;
var filec = 2;
var filed = 3;
var filee = 4;
var filef = 5;
var fileg = 6;
var fileh = 7;

var rank1 = 0;
var rank2 = 1;
var rank3 = 2;
var rank4 = 3;
var rank5 = 4;
var rank6 = 5;
var rank7 = 6;
var rank8 = 7;

var RAYS = [
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
    -15, 0, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, -17
];




function square(file, rank) {
    return Number((rank) * 16 + file);
}

function file(square) {
    return square & 7;
}

function rank(square) {
    return square >>> 4;
}


function squareFromStr(str) {
    return square(fileFromChar(str[0]), rankFromChar(str[1]))
}



function validSquare(sq) {
    return (sq & 0x88) == 0
}

function squareToStr(sq) {
    if (!validSquare(sq)) return '#'
    return fileToChar(file(sq)) + rankToChar(rank(sq))
}



var checkbit = []

checkbit[p] = 1
checkbit[n] = 1 << 1
checkbit[b] = 1 << 2
checkbit[r] = 1 << 3
checkbit[q] = checkbit[b] | checkbit[r]
checkbit[k] = 1 << 5

checkbit[wp] = checkbit[bp] = checkbit[p]
checkbit[wn] = checkbit[bn] = checkbit[n]
checkbit[wb] = checkbit[bb] = checkbit[b]
checkbit[wr] = checkbit[br] = checkbit[r]
checkbit[wq] = checkbit[bq] = checkbit[q]

var attackbit = []
attackbit[p] = attackbit[wp] = attackbit[bp] = 1
attackbit[n] = attackbit[wn] = attackbit[bn] = 1 << 1
attackbit[b] = attackbit[wb] = attackbit[bb] = 1 << 2
attackbit[r] = attackbit[wr] = attackbit[br] = 1 << 3
attackbit[q] = attackbit[wq] = attackbit[bq] = 1 << 4
attackbit[k] = attackbit[wk] = attackbit[bk] = 1 << 5

let piecebyoffset = []
piecebyoffset[16] = piecebyoffset[1] = checkbit[r]
piecebyoffset[17] = piecebyoffset[15] = checkbit[b]





/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Board {

    stm = 0; // side to move
    counter50 = 0;
    castlingRights = 0;
    enpassantSquare = -1;
    movenumber = 0;
    movehalfnumber = 0;

    // Debug flag for verbose output
    debugvalue = true

    moves = []
    movesStr = []

    checkingSquare = -1 // Square from which check is given

    inCheck = false
    inDoubleCheck = false
    inCheckMate = false
    inStalemate = false
    isDraw = false

    pieceat = new Uint8Array(128).fill(0); // 128 por ser 0x88
    kingsquares = [0, 0]
    kingschecks = [0, 0]

    kingescapes = [[], [],]

    piecesquares = [
        new Int8Array(16).fill(-1),
        new Int8Array(16).fill(-1),
    ]

    addpiece(color, piece, square) {

        this.pieceat[square] = piece

        if (ptype(piece) === k) {
            this.kingsquares[color] = square
            return
        }

        for (let i = 0; i < this.piecesquares[color].length; i++) {
            if (this.piecesquares[color][i] == -1) {
                this.piecesquares[color][i] = square
                return
            }
        }

        console.assert(false, 'error in addpiece', color, pieceChar(piece), squareToStr(square))
    }

    movepiece(color, piece, from, to) {

        this.pieceat[to] = this.pieceat[from]
        this.pieceat[from] = empty

        if (ptype(piece) === k) {
            this.kingsquares[color] = to
            return
        }

        for (let i = 0; i < this.piecesquares[color].length; i++) {
            if (this.piecesquares[color][i] == from) {
                this.piecesquares[color][i] = to
                return
            }
        }

        console.assert(false, 'error in movepiece', color, pieceChar(piece), squareToStr(from), squareToStr(to))
    }

    removepiece(color, square) {

        this.pieceat[square] = empty

        for (let i = 0; i < this.piecesquares[color].length; i++) {
            if (this.piecesquares[color][i] == square) {
                this.piecesquares[color][i] = -1
                return
            }
        }

        console.assert(false, 'error in removepiece', color, square)
    }


    attacks = [
        new Uint8Array(128).fill(0), // Bitfield of piece types attacking each square by side
        new Uint8Array(128).fill(0)
    ]

    numattacks = [
        new Uint8Array(128).fill(0), // Number of pieces attacking each square by side
        new Uint8Array(128).fill(0)
    ]

    inchekValidSquares = [
        new Uint8Array(128).fill(0), // Valid squares when in check (blocking or capturing)
        new Uint8Array(128).fill(0)
    ]

    pinDirection = [
        new Int8Array(128).fill(0), // Direction of absolute pins (0 = not pinned)
        new Int8Array(128).fill(0)
    ]

    chekingSquares = [
        new Uint8Array(128).fill(0),  // Squares that could give check and piece type that would check
        new Uint8Array(128).fill(0)
    ]

    matingSquares = [
        new Uint8Array(128).fill(0),  // Squares that could give checkmate and piece type
        new Uint8Array(128).fill(0)
    ]


    addattack(who, piecetype, from, to) {
        this.attacks[who][to] |= attackbit[piecetype]
        this.numattacks[who][to]++

        // Check if this attack gives check to the enemy king
        if (to === this.kingsquares[1 - who]) {

            if ((1 - who) === this.stm) {
                // Normal check detected
                this.kingschecks[1 - who]++
            } else {
                // Illegal position: king can be captured on opponent's turn
            }

            this.checkingSquare = from
            this.inchekValidSquares[1 - who][from] = 1
        }
    }

    addSliderAttack(who, piecetype, from, to, direction) {

        // Called when a sliding piece encounters another piece (friendly or enemy)
        // Detects the piece type and continues the ray for x-ray attacks and pins

        this.attacks[who][to] |= attackbit[piecetype]
        this.numattacks[who][to]++

        let opossingSide = 1 - who
        let enemykingsquare = this.kingsquares[opossingSide]
        // let attackingpiece = this.pieceat[from]
        let attackedpiece = this.pieceat[to]

        // Slider attacks enemy king - this is check
        if (to === enemykingsquare) {
            // Add one more square behind the king in the check direction
            this.attacks[who][to + direction] |= attackbit[piecetype]
            this.numattacks[who][to + direction]++
            this.kingschecks[opossingSide]++

            // Mark squares that block the ray as valid during check
            do {
                to -= direction
                if (!validSquare(to)) break
                this.inchekValidSquares[opossingSide][to] = 1
            } while (to !== from)

            return
        }

        // Encounters a piece that is not the enemy king
        // Check if this could pin a piece against the enemy king
        let difference = from - enemykingsquare;
        let index = difference + 119;
        let offset = RAYS[index];

        if (direction === offset) {
            // The sliding piece is attacking in the direction of the enemy king
            // Possible pinned piece or discovered check

            let piecesinbetween = 0
            let next = to + offset
            while (next !== enemykingsquare) {
                if (!validSquare(next)) break
                if (this.pieceat[next] !== empty) {
                    piecesinbetween += 1
                    // break
                }
                next = next + offset
            }
            // Count pieces between slider and enemy king

            if (piecesinbetween === 0) {
                if (pcolor(attackedpiece) === opossingSide) {
                    this.pinDirection[this.stm][to] = offset
                    // Pin detected and recorded
                } else {
                    // Possible discovered check scenario
                }
            }
        }


        // Increment x-ray attacks if it's a friendly piece that can move in the same direction
        let samecolor = (who === pcolor(attackedpiece))
        let bitsok = checkbit[piecetype] & checkbit[ptype(attackedpiece)] & piecebyoffset[Math.abs(direction)]
        let canxray = samecolor && (bitsok > 0)

        if (canxray) {
            // Continue the ray for x-ray attacks
            // NOTE: This is not double check - see FEN: 1r1qr1k1/5p1p/p2p1B2/1p1pn3/2P1P2P/5P2/PP4Q1/1K4R1 b - - 6 30
            let next = to + direction
            while (validSquare(next)) {
                this.numattacks[who][next]++
                if (this.pieceat[next] !== empty) {
                    break
                }
                next = next + direction
            }
        }

    }

    getMoveStr(move) {

        let checkStr = (move.mask & mask_check) ? '+' : '';
        let discovercheckStr = (move.mask & mask_discovercheck) ? 'D+' : '';

        if (move.mask & mask_castling) {
            let movingStr = (file(move.to) === fileg) ? '00' : '000'
            return movingStr + checkStr + discovercheckStr
        } else {
            // console.log(move)
            let promStr = (move.promotedpiece === 0) ? '' : '=' + charpieces[move.promotedpiece & 7]
            let movingStr = charpieces[move.movingpiece & 7];
            let captureStr = (move.captured) ? 'x' + charpieces[move.captured & 7] : '';

            // let doublecheckStr = (move.mask & mask_doublecheck) ? '+' : '';            
            let enpassantkStr = (move.mask & mask_ep) ? 'ep' : '';
            let safeStr = (move.mask & mask_safe) ? '' : '.';
            let hangingStr = (move.mask & mask_hanging) ? '?' : '';
            let freecaptStr = (move.mask & mask_freecapture) ? '@' : '';
            let winningStr = (move.mask & mask_winningcapture) ? '!' : '';

            return movingStr +
                squareToStr(move.from) +
                captureStr +
                squareToStr(move.to) +
                promStr +
                checkStr + discovercheckStr + enpassantkStr +
                safeStr + hangingStr + winningStr + freecaptStr
        }
    }

    /**
     * Add a move to the move list with tactical information
     * Only adds legal moves - checks pins, check evasions, and tactical flags
     */
    addmove(from, to, maskbits = 0, promotedpiece = 0) {

        let movingpiece = this.pieceat[from]

        // If in check, move must be to a valid square (block or capture checking piece)
        if (this.inCheck && ptype(movingpiece) !== k && this.inchekValidSquares[this.stm][to] === 0) return

        // Check if the piece is pinned
        let pindir = this.pinDirection[this.stm][from]
        if (pindir !== 0) {

            // Attempting to move a pinned piece

            let difference = from - to;
            let index = difference + 119;
            let movedirecction = RAYS[index];

            if (Math.abs(pindir) !== Math.abs(movedirecction)) {
                // Pin confirmed - move is illegal
                return
            }
            // Pin not confirmed - piece can move along pin direction
        }


        let capturedpiece = (maskbits & mask_ep) ? p : this.pieceat[to]
        let chekingpiece = (maskbits & mask_promotion) ? promotedpiece : movingpiece

        // Check if the move gives check
        if (this.chekingSquares[this.stm][to] & checkbit[chekingpiece]) {
            maskbits |= mask_check
        }

        // Check if castling gives check (rook move)
        // Example positions:
        // r1bk1b1r/ppp1qppp/5n2/4p1B1/4P3/2N2N2/PPP2PPP/R3KB1R w KQ - 3 9
        // r6r/ppp2kpp/1b6/2B5/4n1BN/2N5/PPP3PP/R3K2R w KQ - 1 19
        if (maskbits & mask_castling) {
            let rookfile = (file(to) === fileg) ? filef : filed
            let rooksquare = square(rookfile, rank(to))
            if (this.chekingSquares[this.stm][rooksquare] & checkbit[r]) {
                maskbits |= mask_check
            }
        }

        // Check if the move gives discovered check
        if (this.chekingSquares[this.stm][from] & mask_discovercheck) {
            let discoverinfo = this.chekingSquares[this.stm][from]
            let offset = discoverinfo & ~mask_discovercheck
            let moveoffset = Math.abs(from - to)
            // Castling with discovered check: r4bnr/ppp2ppp/n1p5/4P3/3P3q/2N1B3/PPP2P1P/R3K2k w Q - 0 13
            if ((moveoffset !== offset) || (maskbits & mask_castling)) {
                // Discovered check detected
                maskbits |= mask_discovercheck
            }
        }

        // Check if the destination square is safe (not attacked)
        let attackbits = this.attacks[opposite(this.stm)][to]
        if (attackbits === 0) {
            maskbits |= mask_safe
        } else {
            if (lowestSetBit(attackbits) < lowestSetBit(attackbit[movingpiece]))
                maskbits |= mask_hanging
        }

        // Check if capturing a piece and if it's undefended or higher value
        if (capturedpiece) {
            maskbits |= mask_capture
            if (bitCount(attackbits) == 0) maskbits |= mask_freecapture
            if (ptype(capturedpiece) > ptype(movingpiece)) maskbits |= mask_winningcapture
        }

        // Finally store the move with all collected tactical information
        this.moves.push({
            from: from,
            to: to,
            promotedpiece: promotedpiece,
            movingpiece: movingpiece,
            captured: capturedpiece,
            mask: maskbits
        })

    }

    addpawnmove(from, to) {

        // If in check, must be to a valid square
        if (this.inCheck && this.inchekValidSquares[this.stm][to] === 0) return

        // If moving to 8th rank, it's a promotion
        if (rank(to) === rank8) {
            this.addmove(from, to, mask_promotion, wn)
            this.addmove(from, to, mask_promotion, wb)
            this.addmove(from, to, mask_promotion, wr)
            this.addmove(from, to, mask_promotion, wq)
            return
        }

        if (rank(to) === rank1) {
            this.addmove(from, to, mask_promotion, bn)
            this.addmove(from, to, mask_promotion, bb)
            this.addmove(from, to, mask_promotion, br)
            this.addmove(from, to, mask_promotion, bq)
            return
        }

        this.addmove(from, to)
    }


    /**
     * Move history for make/unmake operations
     * Stores all information needed to reverse a move
     */
    history = {
        from: new Uint8Array(MAXPLY).fill(0),
        to: new Uint8Array(MAXPLY).fill(0),
        capturedpiece: new Uint8Array(MAXPLY).fill(0),
        promotedpiece: new Uint8Array(MAXPLY).fill(0),
        counter50: new Uint8Array(MAXPLY).fill(0),
        castlingRights: new Uint8Array(MAXPLY).fill(0),
        enpassantSquare: new Int8Array(MAXPLY).fill(-1),

        movestr: [],

        ply: 0,

        reset: function () {
            this.ply = 0
            this.movestr = []

            this.from.fill(0)
            this.to.fill(0)
            this.capturedpiece.fill(0)
            this.promotedpiece.fill(0)
            this.counter50.fill(0)
            this.castlingRights.fill(0)
            this.enpassantSquare.fill(0)
        },

        add: function (from, to, cp, pm, c50, cr, ep) {

            this.to[this.ply] = to
            this.from[this.ply] = from
            this.counter50[this.ply] = c50
            this.capturedpiece[this.ply] = cp
            this.promotedpiece[this.ply] = pm
            this.castlingRights[this.ply] = cr
            this.enpassantSquare[this.ply] = ep

            this.ply += 1

            this.movestr[this.ply] = squareToStr(from) + squareToStr(to)
        },

        pop: function () {

            if (this.ply == 0) return false

            this.ply -= 1

            return {
                to: this.to[this.ply],
                from: this.from[this.ply],
                counter50: this.counter50[this.ply],
                capturedpiece: this.capturedpiece[this.ply],
                promotedpiece: this.promotedpiece[this.ply],
                castlingRights: this.castlingRights[this.ply],
                enpassantSquare: this.enpassantSquare[this.ply],
            }
        },

        debug: function () {

            // console.log(this)

            let historymoves = []
            for (let ply = 0; ply < this.ply; ply++) {

                if (this.from[ply] === this.to[ply]) break

                historymoves.push(squareToStr(this.from[ply]) + squareToStr(this.to[ply]))
            }

            return historymoves.join(' ')
        }
    }


    constructor() {
        // Board is ready to use after construction
    }

    reset() {
        this.history.reset()

        this.stm = 0; // side to move
        this.counter50 = 0;
        this.castlingRights = 0;
        this.enpassantSquare = -1;
        this.movenumber = 0;
        this.movehalfnumber = 0;

        // enemy = 1 - this.stm
        // numlegalmoves = 0
        this.debugvalue = true

        this.moves = []
        this.movesStr = []

        this.inCheck = false
        this.inDoubleCheck = false
        this.inCheckMate = false
        this.inStalemate = false
        this.isDraw = false

        this.pieceat.fill(0)
        this.kingsquares.fill(0)
        this.piecesquares[white].fill(0)
        this.piecesquares[black].fill(0)

        this.kingescapes = [[], [],]
        this.kingschecks = [0, 0]

        // console.clear()
    }


    /**
     * Generate all legal moves for the current position
     * This is the core algorithm that:
     * 1. Calculates checking squares and discovered checks
     * 2. Detects pins and x-ray attacks
     * 3. Generates only strictly legal moves (no pseudo-moves)
     * 4. Enriches moves with tactical information
     */
    generateMoves() {

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // INITIALIZATION
        /////////////////////////////////////////////////////////////////////////////////////////////////
        this.attacks[white].fill(0)
        this.attacks[black].fill(0)
        this.numattacks[white].fill(0)
        this.numattacks[black].fill(0)
        this.inchekValidSquares[white].fill(0)
        this.inchekValidSquares[black].fill(0)
        this.pinDirection[white].fill(0)
        this.pinDirection[black].fill(0)
        this.chekingSquares[white].fill(0)
        this.chekingSquares[black].fill(0)
        this.matingSquares[white].fill(0)
        this.matingSquares[black].fill(0)

        this.kingescapes = [[], [],]
        this.kingschecks = [0, 0]

        const offsets = []
        offsets[n] = [-18, -33, -31, -14, 18, 33, 31, 14]
        offsets[b] = [-17, -15, 17, 15]
        offsets[r] = [-16, 1, 16, -1]
        offsets[q] = [-17, -16, -15, 1, 17, 16, 15, -1]
        offsets[k] = [-17, -16, -15, 1, 17, 16, 15, -1]

        const piecebits = []
        piecebits[p] = 1
        piecebits[n] = 2
        piecebits[b] = 4
        piecebits[r] = 8

        let to = 0
        let piece = 0
        let color = 0
        let type = 0

        let sq = 0
        let dest = 0
        let csq = 0

        this.moves = []
        this.movesStr = []

        const sign = (this.stm === white) ? 1 : -1
        const secondrank = (this.stm === white) ? rank2 : rank7

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // CALCULATE CHECKING SQUARES
        // For each side, mark squares from which pieces could give check to enemy king
        /////////////////////////////////////////////////////////////////////////////////////////////////
        if (true)
            for (let turncolor = white; turncolor <= black; turncolor++) {

                let enemyking = this.kingsquares[1 - turncolor]
                let offsetsign = (turncolor === white) ? -1 : 1;
                let oppcolor = 1 - turncolor

                // Pawn checks
                for (const offset of [17, 15]) {
                    csq = enemyking + (offset * offsetsign)
                    if (csq & 0x88) continue
                    if (pcolor(this.pieceat[csq]) !== turncolor)
                        this.chekingSquares[turncolor][csq] |= checkbit[p]
                }

                // Knight checks
                for (const offset of offsets[n]) {
                    if (validSquare(csq = (enemyking + offset)))
                        if (pcolor(this.pieceat[csq]) !== turncolor)
                            this.chekingSquares[turncolor][csq] |= checkbit[n]
                }

                // Bishop or rook checks
                for (let piece of [b, r]) {
                    for (const offset of offsets[piece]) {
                        to = enemyking
                        do {
                            to = to + offset
                            if (to & 0x88) break

                            dest = this.pieceat[to]

                            // casilla vacia
                            if (dest === empty) {
                                this.chekingSquares[turncolor][to] |= checkbit[piece]
                                continue
                            }

                            // pieza rival
                            if (pcolor(dest) === oppcolor) {
                                this.chekingSquares[turncolor][to] |= checkbit[piece]
                                break;
                            }

                            // posible jaque a la descubierta
                            let blockingsq = to
                            do {
                                // console.log(squareToStr(to))

                                to = to + offset
                                if (to & 0x88) break

                                dest = this.pieceat[to]
                                if (dest === empty) continue
                                if (pcolor(dest) !== turncolor) break

                                if ((ptype(dest) === piece) || (ptype(dest) === q)) {
                                    this.chekingSquares[turncolor][blockingsq] = Math.abs(offset) | mask_discovercheck

                                    // console.log('marcando como descubierta', squareToStr(blockingsq), squareToStr(to), Math.abs(offset), this.chekingSquares[turncolor][blockingsq])
                                }

                                break

                            } while (true)

                            break

                        } while (true)
                    }
                }
            }


        /////////////////////////////////////////////////////////////////////////////////////////////////
        // ENEMY ATTACKS AND PINNED PIECES
        // Calculate squares attacked by opponent and detect pinned pieces
        /////////////////////////////////////////////////////////////////////////////////////////////////
        let turn = opposite(this.stm)
        let enemysign = sign * - 1

        // for (let sq = a1; sq <= h8; sq++) {
        // let enemypieces = this.piecesquares[turn]
        // console.log ( enemypieces )
        // enemypieces.push(this.kingsquares[turn])

        for (sq of this.piecesquares[turn]) {

            if (sq & 0x88) { sq += 7; continue; }

            piece = this.pieceat[sq]
            if (!piece) continue

            color = pcolor(piece)
            if (color !== turn) continue

            let piecetype = ptype(piece)
            // console.log(squareToStr(sq), piece, color, type)

            switch (piecetype) {
                case p:

                    for (const offset of [17, 15]) {
                        to = sq + (offset * enemysign)
                        if (to & 0x88) continue
                        this.addattack(turn, piecetype, sq, to)
                    }
                    break

                // saltadoras
                case n:
                case k:

                    for (const offset of offsets[piecetype]) {
                        to = sq + offset
                        if (to & 0x88) continue
                        this.addattack(turn, piecetype, sq, to)
                    }
                    break

                // deslizadoras
                case b:
                case r:
                case q:

                    for (const offset of offsets[piecetype]) {
                        to = sq
                        do {
                            to = to + offset
                            if (to & 0x88) break

                            dest = this.pieceat[to]

                            // pieza ocupada
                            if (dest !== empty) {
                                this.addSliderAttack(turn, piecetype, sq, to, offset)
                                break;
                            }

                            // casilla vacia 
                            this.addattack(turn, piecetype, sq, to)
                        } while (true)
                    }

                    break
                default:
                    console.log('Error en tipo de pieza')
                    break
            }
        }

        sq = this.kingsquares[turn]
        for (const offset of offsets[k]) {
            to = sq + offset
            if (to & 0x88) continue

            this.addattack(turn, k, sq, to)
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // CHECK STATUS
        /////////////////////////////////////////////////////////////////////////////////////////////////
        this.inCheck = this.kingschecks[this.stm] > 0
        this.inDoubleCheck = this.kingschecks[this.stm] > 1


        /////////////////////////////////////////////////////////////////////////////////////////////////
        // GENERATE MOVES FOR SIDE TO MOVE
        // If not in double check, generate moves for all pieces except king
        /////////////////////////////////////////////////////////////////////////////////////////////////
        if (!this.inDoubleCheck)
            for (sq of this.piecesquares[this.stm]) {

                if (sq & 0x88) { sq += 7; continue; }

                piece = this.pieceat[sq]
                if (!piece) continue

                color = pcolor(piece)
                if (color !== this.stm) continue

                type = ptype(piece)
                // console.log(squareToStr(sq), piece, color, type)

                switch (type) {
                    case p:

                        // Pawn advances
                        if (this.pieceat[to = (sq + (16 * sign))] === empty) {
                            this.addpawnmove(sq, to)

                            if (rank(sq) === secondrank) {
                                if (this.pieceat[to = (sq + (32 * sign))] === empty)
                                    this.addmove(sq, to)
                            }
                        }

                        // Pawn captures
                        for (const offset of [17, 15]) {
                            if (!validSquare(to = (sq + (offset * sign)))) continue
                            this.addattack(this.stm, type, sq, to)

                            if (pcolor(this.pieceat[to]) === opposite(this.stm)) {
                                this.addpawnmove(sq, to)
                            }
                            if (to === this.enpassantSquare) {
                                // En passant capture - check for horizontal pin
                                // Test positions:
                                // 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
                                // 8/1K2p3/8/3P4/8/5b2/6k1/8 b - - 0 1
                                let ilegalep = false
                                let kingsquare = this.kingsquares[this.stm]
                                if (rank(sq) === rank(kingsquare)) {
                                    // Check for horizontal pin: 7k/8/8/K1pP3r/8/8/8/8 w - c6 0 1
                                    let dir = (file(kingsquare) < file(sq)) ? 1 : -1
                                    let next = kingsquare + dir
                                    let count = 0
                                    while (validSquare(next)) {

                                        let piece = this.pieceat[next]
                                        next = next + dir

                                        if (piece === empty) continue

                                        count++
                                        if ((count === 3) && ((ptype(piece) === r) | (ptype(piece) === q))
                                            && (pcolor(piece) === opposite(this.stm))) {
                                            // Illegal en passant confirmed (horizontal pin)
                                            ilegalep = true
                                            break
                                        }
                                    }
                                }

                                if (!ilegalep) {
                                    // If check is given by a pawn that can be captured en passant,
                                    // capturing it is a valid move
                                    // Example: 8/8/3p4/1Pp4r/1K3pk1/8/4P1P1/1R6 w - c6 0 3
                                    if (this.inCheck) {
                                        let chekingpiece = this.pieceat[this.checkingSquare]
                                        if (ptype(chekingpiece) === p) {
                                            // Mark en passant square as valid
                                            this.inchekValidSquares[this.stm][this.enpassantSquare] = 1
                                            this.addmove(sq, to, mask_ep)
                                            // Unmark it so other pieces don't go there
                                            this.inchekValidSquares[this.stm][this.enpassantSquare] = 0
                                        }
                                    } else {
                                        this.addmove(sq, to, mask_ep)
                                    }
                                }
                            }


                        }

                        break

                    // Jumping pieces
                    case n:

                        if (this.pinDirection[this.stm][sq] !== 0) break // Pinned knight can never move

                        for (const offset of offsets[type]) {
                            to = sq + offset

                            if (to & 0x88) continue

                            dest = this.pieceat[to]
                            this.addattack(this.stm, type, sq, to)

                            if ((dest === empty) || (pcolor(dest) === opposite(this.stm)))
                                this.addmove(sq, to)
                        }

                        break

                    // Sliding pieces
                    case b:
                    case r:
                    case q:

                        let pindir = this.pinDirection[this.stm][sq]

                        for (const offset of offsets[type]) {

                            if ((pindir > 0) && (Math.abs(pindir) !== Math.abs(offset))) continue

                            to = sq
                            do {
                                to = to + offset
                                if (to & 0x88) break

                                dest = this.pieceat[to]


                                if (dest !== empty) {
                                    if (pcolor(dest) === opposite(this.stm)) {
                                        this.addmove(sq, to);
                                    }
                                    this.addSliderAttack(this.stm, type, sq, to, offset)
                                    break;
                                }

                                this.addattack(this.stm, type, sq, to)
                                this.addmove(sq, to);
                            } while (true)
                        }

                        break
                    default:
                        // console.log('Error en tipo de pieza', type)
                        break
                }
            }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // KING MOVES - ONLY MOVES AVAILABLE IN DOUBLE CHECK
        // Generate moves for our king (always generated, even in double check)
        /////////////////////////////////////////////////////////////////////////////////////////////////
        sq = this.kingsquares[this.stm]
        for (const offset of offsets[k]) {
            to = sq + offset
            if (to & 0x88) continue

            this.addattack(this.stm, k, sq, to)

            if (this.numattacks[opposite(this.stm)][to] > 0) continue

            dest = this.pieceat[to]
            if ((dest === empty) || pcolor(dest) === opposite(this.stm)) {
                this.addmove(sq, to)
                // this.kingescapes[this.stm].push(Math.abs(offset))
                let absoffset = Math.abs(offset)
                const found = this.kingescapes[this.stm].find(element => element === absoffset)
                if (!found) this.kingescapes[this.stm].push(absoffset)
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // MARK ENEMY KING ESCAPE SQUARES TO DETECT MATING SQUARES
        /////////////////////////////////////////////////////////////////////////////////////////////////


        sq = this.kingsquares[opposite(this.stm)]
        for (const offset of offsets[k]) {
            to = sq + offset
            if (to & 0x88) continue

            // Square the king wants to move to is not attacked
            if (this.numattacks[this.stm][to] > 0) {
                // Check for queen contact check

                if ((this.numattacks[this.stm][to] > 1) &&
                    (this.numattacks[opposite(this.stm)][to] === 1) &&
                    (this.attacks[this.stm][to] & attackbit[q])
                ) {
                    this.matingSquares[this.stm][to] = this.chekingSquares[this.stm][to]
                }
                continue
            }

            dest = this.pieceat[to]
            // King can move to empty square or capture our piece
            if ((dest === empty) || pcolor(dest) === this.stm) {
                let absoffset = Math.abs(offset)
                const found = this.kingescapes[opposite(this.stm)].find(element => element === absoffset)
                if (!found) this.kingescapes[opposite(this.stm)].push(absoffset)
            }
        }

        if (this.kingescapes[opposite(this.stm)].length === 0) {
            // No escape squares - all safe checks would be (almost) mate
            for (let i = 0; i < 128; i++) {

                if (!validSquare(i)) continue
                let numattackers = this.numattacks[this.stm][i]
                if (numattackers === 0) continue

                let numdefenders = this.numattacks[opposite(this.stm)][i]

                // If number of defenders is zero and not occupied by our piece
                // Can be empty or occupied by undefended enemy piece
                if ((numdefenders === 0) && (pcolor(this.pieceat[i]) !== this.stm)) {
                    if (this.chekingSquares[this.stm][i] & this.attacks[this.stm][i]) {
                        this.matingSquares[this.stm][i] = this.chekingSquares[this.stm][i]
                    }
                    continue
                }

                // If defenders is 1, attackers > 1, and only defender is the king
                if ((numdefenders === 1) && (numattackers > 1)) {
                    let kingistheonlydefender = this.attacks[opposite(this.stm)][i] === attackbit[k]

                    if (kingistheonlydefender) {
                        this.matingSquares[this.stm][i] = this.chekingSquares[this.stm][i]
                        continue
                    }
                }
            }
        } else if (this.kingescapes[opposite(this.stm)].length === 1) {
            // King has only one escape direction
            let offset = this.kingescapes[opposite(this.stm)][0]

            for (let offset2 of [offset, -offset]) {
                to = sq + offset2
                while (validSquare(to)) {
                    dest = this.pieceat[to]

                    if (this.numattacks[this.stm][to] > this.numattacks[opposite(this.stm)][to]) {
                        this.matingSquares[this.stm][to] = this.chekingSquares[this.stm][to]
                    }

                    if ((dest === empty) || pcolor(dest) === opposite(this.stm)) {
                        // Possible mate square
                    }
                    if (dest !== 0) break
                    to = to + offset2
                }
            }

        }



        /////////////////////////////////////////////////////////////////////////////////////////////////
        // CASTLING
        /////////////////////////////////////////////////////////////////////////////////////////////////
        if (!this.inCheck) {
            sq = this.kingsquares[this.stm]
            if (this.stm === white) {
                if (sq === e1) {
                    if ((this.pieceat[f1] === empty) &&
                        (this.numattacks[opposite(this.stm)][f1] === 0) &&
                        (this.numattacks[opposite(this.stm)][g1] === 0) &&
                        (this.pieceat[g1] === empty) &&
                        (this.pieceat[h1] === wr) &&
                        (this.castlingRights & whitecancastleK)
                    ) this.addmove(e1, g1, mask_castling)

                    if ((this.pieceat[d1] === empty) &&
                        (this.numattacks[opposite(this.stm)][d1] === 0) &&
                        (this.numattacks[opposite(this.stm)][c1] === 0) &&
                        (this.pieceat[c1] === empty) &&
                        (this.pieceat[b1] === empty) &&
                        (this.pieceat[a1] === wr) &&
                        (this.castlingRights & whitecancastleQ)
                    ) this.addmove(e1, c1, mask_castling)
                }
            } else {
                if (sq === e8) {
                    if ((this.pieceat[f8] === empty) &&
                        (this.numattacks[opposite(this.stm)][f8] === 0) &&
                        (this.numattacks[opposite(this.stm)][g8] === 0) &&
                        (this.pieceat[g8] === empty) &&
                        (this.pieceat[h8] === br) &&
                        (this.castlingRights & blackcancastleK)
                    ) this.addmove(e8, g8, mask_castling)

                    if ((this.pieceat[d8] === empty) &&
                        (this.numattacks[opposite(this.stm)][d8] === 0) &&
                        (this.numattacks[opposite(this.stm)][c8] === 0) &&
                        (this.pieceat[c8] === empty) &&
                        (this.pieceat[b8] === empty) &&
                        (this.pieceat[a8] === br) &&
                        (this.castlingRights & blackcancastleQ)
                    ) this.addmove(e8, c8, mask_castling)
                }
            }
        }


        /////////////////////////////////////////////////////////////////////////////////////////////////
        // GAME STATE CHECK (commented out for performance)
        /////////////////////////////////////////////////////////////////////////////////////////////////

    }

    /**
     * Undo the last move made
     * Restores board state and piece positions
     */
    undomove() {
        let undo = this.history.pop()

        if (undo == false) return false;

        this.stm = opposite(this.stm)
        let signo = (this.stm === white) ? 1 : -1

        if (undo.promotedpiece !== 0) {
            this.pieceat[undo.to] = (this.stm == white) ? wp : bp;
        }

        let movingpiece = this.pieceat[undo.to]

        this.movepiece(this.stm, movingpiece, undo.to, undo.from)

        if (undo.capturedpiece) {
            this.addpiece(opposite(this.stm), undo.capturedpiece, undo.to)
        }
        console.assert(this.stm === pcolor(movingpiece), 'undomove() piece color does not match side to move')

        this.counter50 = undo.counter50
        this.castlingRights = undo.castlingRights
        this.enpassantSquare = undo.enpassantSquare



        if (undo.to === undo.enpassantSquare) {
            if (ptype(movingpiece) === p) {
                // Restore captured pawn from en passant
                this.addpiece(opposite(this.stm), (opposite(this.stm) === white) ? wp : bp, undo.enpassantSquare - (16 * signo))
            }
        }

        // Undo castling - move rook back
        if ((undo.from == e1) && (undo.to == g1) && (movingpiece == wk)) {
            console.assert(this.pieceat[f1] == wr, 'no rook on f1 when undoing')
            this.movepiece(this.stm, wr, f1, h1)
        }

        if ((undo.from == e1) && (undo.to == c1) && (movingpiece == wk)) {
            console.assert(this.pieceat[d1] == wr, 'no rook on d1 when undoing')
            this.movepiece(this.stm, wr, d1, a1)
        }

        if ((undo.from == e8) && (undo.to == g8) && (movingpiece == bk)) {
            console.assert(this.pieceat[f8] == br, 'no rook on f8 when undoing')
            this.movepiece(this.stm, wr, f8, h8)
        }

        if ((undo.from == e8) && (undo.to == c8) && (movingpiece == bk)) {
            console.assert(this.pieceat[d8] == br, 'no rook on d8 when undoing')
            this.movepiece(this.stm, wr, d8, a8)
        }

        this.movehalfnumber -= 1
        this.movenumber = this.movehalfnumber / 2 >> 0 // Convert to integer

        return true;
    }

    /**
     * Make a move on the board
     * Updates board state, castling rights, en passant, and move counters
     */
    makemove_old(from, to, promotedpiece = 0) {

        let capturedpiece = this.pieceat[to]

        this.history.add(from, to, capturedpiece, promotedpiece, this.counter50, this.castlingRights, this.enpassantSquare)

        let iscapture = this.pieceat[to] != empty
        let movingpiece = this.pieceat[from]

        console.assert(movingpiece != empty)
        if (movingpiece === empty) {
            debugger
        }

        let ispawnmove = movingpiece == wp || movingpiece == bp
        let ispawnpush = !iscapture && ispawnmove && Math.abs(to - from) == 32
        let ispromotion = ispawnmove && (rank(to) == rank8 || rank(to) == rank1)
        let isenpassant = ispawnmove && to == this.enpassantSquare

        console.assert(!ispromotion || promotedpiece > 0, 'promotion without promoted piece')

        this.movehalfnumber += 1
        this.movenumber = this.movehalfnumber / 2 >> 0 // Convert to integer

        if (iscapture || ispawnmove) {
            this.counter50 = 0
        } else {
            this.counter50 += 1
        }

        // En passant square
        this.enpassantSquare = -1
        if (ispawnpush) {
            let enemypawn = (this.stm === white) ? bp : wp;

            if ((this.pieceat[to + 1] === enemypawn) || (this.pieceat[to - 1] === enemypawn)) {
                this.enpassantSquare = (from + to) / 2
            }

            console.assert(
                (this.stm == white && rank(from) == rank2 && rank(to) == rank4) ||
                (this.stm == black && rank(from) == rank7 && rank(to) == rank5),
                'error generating en passant square'
            )
        }

        if ((from == a1) || (to == a1)) this.castlingRights &= ~whitecancastleQ
        if ((from == h1) || (to == h1)) this.castlingRights &= ~whitecancastleK
        if ((from == a8) || (to == a8)) this.castlingRights &= ~blackcancastleQ
        if ((from == h8) || (to == h8)) this.castlingRights &= ~blackcancastleK

        // White castling
        if ((from == e1) && (movingpiece == wk)) {
            if (to == g1) {
                console.assert(this.castlingRights && whitecancastleK, 'whitecancastleK')
                console.assert(this.pieceat[h1] == wr, 'no rook on h1 when castling')
                this.movepiece(this.stm, wr, h1, f1)
            }
            if (to == c1) {
                console.assert(this.castlingRights && whitecancastleQ, 'whitecancastleQ')
                console.assert(this.pieceat[a1] == wr, 'no rook on a1 when castling')
                this.movepiece(this.stm, wr, a1, d1)
            }
            this.castlingRights &= ~(whitecancastleQ | whitecancastleK)
        }

        // Black castling
        if ((from == e8) && (movingpiece == bk)) {
            if (to == g8) {
                console.assert(this.castlingRights && blackcancastleK, 'blackcancastleK')
                console.assert(this.pieceat[h8] == br, 'no rook on h8 when castling')
                this.movepiece(this.stm, br, h8, f8)
            }
            if (to == c8) {
                console.assert(this.castlingRights && blackcancastleK, 'blackcancastleK')
                console.assert(this.pieceat[a8] == br, 'no rook on a8 when castling')
                this.movepiece(this.stm, br, a8, d8)
            }
            this.castlingRights &= ~(blackcancastleQ | blackcancastleK)
        }

        if (iscapture) {
            this.removepiece(opposite(this.stm), to)
        }

        this.movepiece(this.stm, movingpiece, from, to)

        if (ispromotion) {
            console.assert(promotedpiece != empty, 'no piece to promote to')
            if ((this.stm == white) && (promotedpiece > 8)) promotedpiece -= 8
            this.pieceat[to] = promotedpiece
        }

        if (isenpassant) {
            let sqpawn = (this.stm == white) ? to - 16 : to + 16;
            this.removepiece(opposite(this.stm), sqpawn)
        }

        this.stm = opposite(this.stm)

        return true
    }

    makemove(move) {
        this.makemove_old(move.from, move.to, move.promotedpiece)
    }

    squareToStr(sq) {
        return squareToStr(sq)
    }

    squareFromStr(str) {
        return squareFromStr(str)
    }

    debug_tableAttacks(side) {

        let attackedsquares = []
        for (let sq = a1; sq <= h8; sq++) {
            let maskAttacks = this.attacks[side][sq]
            if (maskAttacks !== 0) {
                console.assert(this.numattacks[side][sq] > 0, ' error debug attacks', squareToStr(sq))
                let attacksStr = ''
                if (maskAttacks & attackbit[p]) attacksStr += 'P'
                if (maskAttacks & attackbit[n]) attacksStr += 'N'
                if (maskAttacks & attackbit[b]) attacksStr += 'B'
                if (maskAttacks & attackbit[r]) attacksStr += 'R'
                if (maskAttacks & attackbit[q]) attacksStr += 'Q'
                if (maskAttacks & attackbit[k]) attacksStr += 'K'
                //  + bitCount(maskAttacks) + ' ' 
                attackedsquares.push(squareToStr(sq) + ' ' + this.numattacks[side][sq] + ' ' + attacksStr)
            }
        }

        console.log('tableAttacks', side, attackedsquares.join(' / '))
    }

    debug_checkingSquares(side) {
        let chekingSquares = []
        for (let sq = a1; sq <= h8; sq++) {

            let checkmask = this.chekingSquares[side][sq]
            let attacksStr = ''

            if (checkmask !== 0) {
                if (checkmask & mask_discovercheck) {
                    chekingSquares.push(squareToStr(sq) + ' D+')
                } else {
                    if (checkmask & checkbit[p]) attacksStr += 'P'
                    if (checkmask & checkbit[n]) attacksStr += 'N'
                    if (checkmask & checkbit[b]) attacksStr += 'B'
                    if (checkmask & checkbit[r]) attacksStr += 'R'
                    if (checkmask & checkbit[q]) attacksStr += 'Q'
                    if (checkmask & checkbit[k]) attacksStr += 'K'

                    chekingSquares.push(squareToStr(sq) + ' ' + attacksStr)
                }
            }
        }

        let safechekingSquares = []
        for (let sq = a1; sq <= h8; sq++) {

            let checkmask = this.chekingSquares[side][sq]
            let attacksStr = ''

            if (checkmask !== 0) {
                if (checkmask & mask_discovercheck) {
                    chekingSquares.push(squareToStr(sq) + ' D+')
                } else {
                    // console.log('scsq', squareToStr(sq),  this.numattacks[side][sq], this.numattacks[opposite(side)][sq] )  
                    if (this.numattacks[side][sq] === 0) continue
                    if (this.numattacks[side][sq] < this.numattacks[opposite(side)][sq]) continue

                    let kingistheonlydefender = this.attacks[opposite(side)][sq] === attackbit[k]
                    if ((this.numattacks[opposite(side)][sq] === 1) && !kingistheonlydefender) continue
                    let ispawncheck = (this.chekingSquares[this.stm][sq] & attackbit[p]) != 0
                    let ourpieceattacks = (this.chekingSquares[this.stm][sq] & this.attacks[this.stm][sq]) != 0

                    // console.log('scsq', squareToStr(sq),  this.numattacks[side][sq], this.numattacks[opposite(side)][sq], ispawncheck, ourpieceattacks )  
                    if (!ourpieceattacks && !ispawncheck) continue

                    // checkmask &= this.attacks[side][sq]
                    // if (checkmask == 0) continue

                    if (checkmask & checkbit[p]) attacksStr += 'P'
                    if (checkmask & checkbit[n]) attacksStr += 'N'
                    if (checkmask & checkbit[b]) attacksStr += 'B'
                    if (checkmask & checkbit[r]) attacksStr += 'R'
                    if (checkmask & checkbit[q]) attacksStr += 'Q'
                    if (checkmask & checkbit[k]) attacksStr += 'K'

                    safechekingSquares.push(squareToStr(sq) + ' ' + attacksStr)
                }
            }
        }

        console.log('chekingSquares', side, chekingSquares.join(' / '))
        console.log('safechekingSquares', side, safechekingSquares.join(' / '))
    }

    debug_matingSquares(side) {
        let matingSquares = []
        for (let sq = a1; sq <= h8; sq++) {

            let checkmask = this.matingSquares[side][sq]
            let attacksStr = ''

            if (checkmask !== 0) {
                if (checkmask & mask_discovercheck) {
                    matingSquares.push(squareToStr(sq) + ' D+')
                } else {
                    if (checkmask & checkbit[p]) attacksStr += 'P'
                    if (checkmask & checkbit[n]) attacksStr += 'N'
                    if (checkmask & checkbit[b]) attacksStr += 'B'
                    if (checkmask & checkbit[r]) attacksStr += 'R'
                    if (checkmask & checkbit[q]) attacksStr += 'Q'
                    if (checkmask & checkbit[k]) attacksStr += 'K'
                    matingSquares.push(squareToStr(sq) + ' ' + attacksStr)
                }
            }
        }

        console.log('matingSquares', side, matingSquares.join(' / '))
    }

    debug_pins(side) {

        let pinSquares = []
        for (let sq = a1; sq <= h8; sq++) {
            let pdir = this.pinDirection[side][sq]
            if (pdir !== 0) pinSquares.push(squareToStr(sq) + ' ' + pdir)
        }

        console.log('pins', side, pinSquares.join(' / '))
    }

    debug_kingSquares() {
        // console.clear()
        // console.log('stm', this.stm)
        // console.log(this.kingsquares)
        console.log('kingsquares', squareToStr(this.kingsquares[white]), squareToStr(this.kingsquares[black]))
    }

    debug_kingescapes(side) {
        console.log('kingescapes', side, this.kingescapes[side])
    }

    debug_pieceSquares() {
        let output = ''
        this.piecesquares[white].forEach(element => {
            output += squareToStr(element) + ' '
        });
        console.log('piecesquares', white, output)

        output = ''
        this.piecesquares[black].forEach(element => {
            output += squareToStr(element) + ' '
        });
        console.log('piecesquares', black, output)
    }

    debug_legalmoves() {
        let moveStr = ''
        for (const move of this.moves) {
            moveStr += this.getMoveStr(move) + ' '
        }

        console.log('legal moves', this.moves.length, moveStr)


    }

    debug_state() {
        if (this.moves.length === 0) {
            if (this.inCheck) {
                this.inCheckMate = true
                this.isDraw = false
                console.log('Checkmate')
            } else {
                this.isDraw = this.inStalemate = true
                console.log('Stalemate')
            }
        } else {
            if (this.inCheck) console.log('check')
            if (this.inDoubleCheck) console.log('double check')
        }
    }



    debug() {

        if (this.debugvalue) {

            console.log('History', this.history.debug())
            this.generateMoves()


            // this.debug_tableAttacks(white)
            // this.debug_tableAttacks(black)

            // this.debug_checkingSquares(white)
            // this.debug_checkingSquares(black)

            // this.debug_matingSquares(white)
            // this.debug_matingSquares(black)

            // this.debug_kingescapes(white)
            // this.debug_kingescapes(black)

            // this.debug_pins(white)
            // this.debug_pins(black)            
            // this.debug_pieceSquares()        
            this.debug_legalmoves()

        }
    }

    setdebug(debug) {
        console.log('setdebug', debug)

        this.debugvalue = [1, '1', 'true', 'on', true].includes(debug)
    }

    /**
     * Performance test - count nodes at given depth
     * Used for move generation validation
     */
    perft(depth) {

        this.generateMoves()
        var moves = this.moves
        var nodes = 0

        var nmoves = moves.length
        if (depth === 1) return nmoves

        for (var i = 0; i < nmoves; i++) {
            this.makemove(moves[i]);
            nodes += this.perft(depth - 1);
            this.undomove();
        }

        return nodes;
    }

    /**
     * Divide - perft with move breakdown
     * Shows node count for each root move
     */
    divide(depth) {

        this.generateMoves()
        var moves = this.moves
        var nodes = 0

        var nmoves = moves.length
        if (depth === 1) return nmoves

        for (var i = 0; i < nmoves; i++) {
            this.makemove(moves[i]);
            let movenodes = this.perft(depth - 1)
            nodes += movenodes
            console.log(this.getMoveStr(moves[i]), movenodes)
            this.undomove();
        }

        return nodes;
    }


    /**
     * Load a position from FEN string
     * Format: pieces side castling enpassant halfmove fullmove
     */
    loadFEN(fenstring) {

        this.reset()

        let file = filea;
        let rank = rank8;
        let [fenboard, fenstm, fencastling, fenep, fen50, fenmn] = fenstring.split(' ');

        // Parse board position
        this.pieceat.fill(empty);
        this.piecesquares[white].fill(-1)
        this.piecesquares[black].fill(-1)

        for (let i = 0; i < fenboard.length; i++) {
            let char = fenboard[i];
            if (char === '/') {
                rank--;
                file = filea;
                continue;
            }

            if (!isNaN(char)) {
                file += Number(char);
                continue;
            }

            let piece = charpieces.findIndex(el => (el === char));
            let sq = square(file, rank)

            if (piece > 0) {
                this.addpiece(pcolor(piece), piece, sq)
                file++;
            }

            console.assert(piece > 0, ' piece > 0 ')
        }

        // Parse side to move
        this.stm = (fenstm === 'w') ? white : black;

        // Parse castling rights
        this.castlingRights = 0
        if (fencastling != '-') {
            for (let i = 0; i < fencastling.length; i++) {
                let char = fencastling[i];
                if (char == 'Q') this.castlingRights |= whitecancastleQ;
                if (char == 'K') this.castlingRights |= whitecancastleK;
                if (char == 'q') this.castlingRights |= blackcancastleQ;
                if (char == 'k') this.castlingRights |= blackcancastleK;
            }
        }

        // Parse en passant square
        this.enpassantSquare = (fenep == '-') ? -1 : squareFromStr(fenep)

        // Parse 50-move counter
        this.counter50 = (fen50 == '-') ? 0 : Number(fen50)

        // Parse move number
        this.movenumber = (fenmn == '-') ? 1 : Number(fenmn)
        if (this.movenumber == 0) this.movenumber = 1

        this.movehalfnumber = ((this.movenumber) * 2) + this.stm
    }

    getFEN() {

        function getFenBoard(board) {

            let fenboard = ''
            let empty = 0

            for (let rank = rank8; rank >= rank1; rank--) {

                for (let file = filea; file <= fileh; file++) {

                    let sq = square(file, rank)
                    let piece = board[sq]

                    if (!piece) { empty++; continue }
                    if (empty > 0) { fenboard += empty.toString(); empty = 0 }

                    fenboard += pieceChar(piece) // charpieces[piece]                                            
                }

                if (empty > 0) { fenboard += empty.toString(); empty = 0 }
                if (rank > rank1) fenboard += '/'
            }

            return fenboard
        }

        function getFenCastling(castlingRights) {

            if (!castlingRights) return '-'

            let cr = ''
            if (castlingRights & whitecancastleK) cr += 'K'
            if (castlingRights & whitecancastleQ) cr += 'Q'
            if (castlingRights & blackcancastleK) cr += 'k'
            if (castlingRights & blackcancastleQ) cr += 'q'

            return cr
        }

        let fenboard = getFenBoard(this.pieceat)
        let fenstm = (this.stm == white) ? 'w' : 'b'
        let fencastling = getFenCastling(this.castlingRights)
        let fenep = (this.enpassantSquare > 0) ? squareToStr(this.enpassantSquare) : '-'
        let fen50 = this.counter50
        let fenmn = this.movenumber

        return [fenboard, fenstm, fencastling, fenep, fen50, fenmn].join(' ')
    }
}

// Node.js export compatibility
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Board };
}
