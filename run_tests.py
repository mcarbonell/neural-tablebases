"""
Suite de pruebas para verificar el estado del proyecto.
Cubre: encoding, flip_board, canonical forms, models, datasets, unranking.
"""
import sys
import os
sys.path.insert(0, 'src')

import chess
import numpy as np
import torch

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))
        FAIL += 1

# ─────────────────────────────────────────────
print("=" * 60)
print("1. ENCODING DIMENSIONS")
print("=" * 60)
from generate_datasets import encode_board, encode_board_relative, flip_board

def make_board(*pieces_squares):
    b = chess.Board(None)
    for piece, sq in pieces_squares:
        b.set_piece_at(sq, piece)
    b.turn = chess.WHITE
    return b

WK = chess.Piece(chess.KING, chess.WHITE)
WQ = chess.Piece(chess.QUEEN, chess.WHITE)
WR = chess.Piece(chess.ROOK, chess.WHITE)
WP = chess.Piece(chess.PAWN, chess.WHITE)
BK = chess.Piece(chess.KING, chess.BLACK)
BP = chess.Piece(chess.PAWN, chess.BLACK)

b3 = make_board((WK, chess.E1), (WQ, chess.D1), (BK, chess.E8))
b4 = make_board((WK, chess.E1), (WR, chess.D1), (BK, chess.E8), (WP, chess.D4))
b4p = make_board((WK, chess.E1), (WP, chess.D4), (BK, chess.E8), (BP, chess.D5))

check("v1 3-piece = 43", encode_board(b3, relative=True).shape[0] == 43)
check("v4 3-piece = 45", encode_board(b3, relative='v4').shape[0] == 45)
check("v1 4-piece = 65", encode_board(b4, relative=True).shape[0] == 65)
check("v4 4-piece = 68", encode_board(b4, relative='v4').shape[0] == 68)
check("v2.1 3-piece = 64", encode_board(b3, relative=True, use_move_distance=True).shape[0] == 64)
check("v2.1 4-piece = 107", encode_board(b4, relative=True, use_move_distance=True).shape[0] == 107)
check("compact 3-piece = 192", encode_board(b3, compact=True, relative=False).shape[0] == 192)

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("2. FLIP_BOARD")
print("=" * 60)

b_black = make_board((WK, chess.E1), (BK, chess.E8))
b_black.turn = chess.BLACK
flipped = flip_board(b_black)

check("flip turno -> WHITE", flipped.turn == chess.WHITE)
check("flip WK existe", len(list(flipped.pieces(chess.KING, chess.WHITE))) == 1)
check("flip BK existe", len(list(flipped.pieces(chess.KING, chess.BLACK))) == 1)

# flip involutivo
ff = flip_board(flipped)
check("flip involutivo (turno)", ff.turn == b_black.turn)
wk_orig = list(b_black.pieces(chess.KING, chess.WHITE))[0]
wk_ff   = list(ff.pieces(chess.KING, chess.WHITE))[0]
check("flip involutivo (posicion WK)", wk_orig == wk_ff)

# v4: BLACK-to-move debe dar mismo encoding que flip+WHITE
b_w = make_board((WK, chess.E1), (WQ, chess.D1), (BK, chess.E8))
b_w.turn = chess.WHITE
b_b = make_board((WK, chess.E1), (WQ, chess.D1), (BK, chess.E8))
b_b.turn = chess.BLACK
enc_w = encode_board(b_w, relative='v4')
enc_b = encode_board(b_b, relative='v4')
# v4 normaliza internamente, pero los colores se invierten -> no deben ser iguales
# Lo importante es que ambos producen 45 dims sin error
check("v4 BLACK-to-move no crashea", enc_b.shape[0] == 45)

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("3. CANONICAL FORMS")
print("=" * 60)
from canonical_forms import (find_canonical_form, board_to_canonical_key,
                              is_canonical, get_symmetries, rotate_board,
                              reflect_board_horizontal)

# 8 simetrías dihedral -> 1 clave canónica
syms = get_symmetries(b3, mode='dihedral')
check("dihedral produce 8 simetrias", len(syms) == 8)
keys = set(board_to_canonical_key(find_canonical_form(s, lambda x: (), mode='dihedral')[0]) for s in syms)
check("8 simetrias -> 1 clave canonica", len(keys) == 1)

# is_canonical es consistente
cf, _ = find_canonical_form(b3, lambda x: (), mode='dihedral')
check("is_canonical(canonical_form) = True", is_canonical(cf, mode='dihedral'))

# auto mode con peones -> file_mirror (2 simetrías)
b_pawn = make_board((WK, chess.A1), (WP, chess.A2), (BK, chess.H8))
syms_p = get_symmetries(b_pawn, mode='auto')
check("auto+pawns -> file_mirror (2 simetrias)", len(syms_p) == 2)

# auto mode sin peones -> dihedral (8 simetrías)
syms_np = get_symmetries(b3, mode='auto')
check("auto sin pawns -> dihedral (8 simetrias)", len(syms_np) == 8)

# file_mirror: 2 simetrías -> 1 clave
syms_fm = get_symmetries(b3, mode='file_mirror')
keys_fm = set(board_to_canonical_key(find_canonical_form(s, lambda x: (), mode='file_mirror')[0]) for s in syms_fm)
check("file_mirror 2 simetrias -> 1 clave", len(keys_fm) == 1)

# rotate_board: rotación 0 = identidad
rot0 = rotate_board(b3, 0)
check("rotate 0 = identidad", board_to_canonical_key(rot0) == board_to_canonical_key(b3))

# rotate_board: 4 rotaciones -> misma clave canónica
rot_keys = set(board_to_canonical_key(find_canonical_form(rotate_board(b3, r), lambda x: (), mode='dihedral')[0]) for r in range(4))
check("4 rotaciones -> 1 clave canonica", len(rot_keys) == 1)

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("4. MODELS")
print("=" * 60)
from models import MLP, SIREN, get_model_for_endgame

# MLP forward pass
mlp3 = get_model_for_endgame('mlp', 3, input_size=43)
x3 = torch.randn(4, 43)
wdl, dtz = mlp3(x3)
check("MLP 3-piece forward: wdl shape", wdl.shape == (4, 3))
check("MLP 3-piece forward: dtz shape", dtz.shape == (4, 1))

mlp4 = get_model_for_endgame('mlp', 4, input_size=68)
x4 = torch.randn(4, 68)
wdl4, dtz4 = mlp4(x4)
check("MLP 4-piece forward: wdl shape", wdl4.shape == (4, 3))
check("MLP 4-piece forward: dtz shape", dtz4.shape == (4, 1))

# SIREN forward pass
siren3 = get_model_for_endgame('siren', 3, input_size=43)
wdl_s, dtz_s = siren3(x3)
check("SIREN 3-piece forward: wdl shape", wdl_s.shape == (4, 3))
check("SIREN 3-piece forward: dtz shape", dtz_s.shape == (4, 1))

# 5-class
mlp5c = get_model_for_endgame('mlp', 3, num_wdl_classes=5, input_size=43)
wdl5, _ = mlp5c(x3)
check("MLP 5-class wdl shape", wdl5.shape == (4, 5))

# param count razonable
params3 = sum(p.numel() for p in mlp3.parameters())
params4 = sum(p.numel() for p in mlp4.parameters())
check("MLP 3-piece params > 100K", params3 > 100_000)
check("MLP 4-piece params > MLP 3-piece", params4 > params3)

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("5. DATASET LOADING")
print("=" * 60)
from train import TablebaseDataset

# NOTA: data/smoke/KPvK.npz fue generado con --version 4 (45 dims, v4)
# El dataset KPvK_canonical.npz en data/ usa v1 (43 dims)
# Ambos son correctos segun como fueron generados
smoke_files = {
    'data/smoke/KQvK.npz': (43, 3, 1),
    'data/smoke/KPvKP.npz': (68, 4, 4),
    'data/smoke/KPvK.npz': (45, 3, 4),   # generado con v4
}
for path, (expected_input, expected_pieces, expected_enc) in smoke_files.items():
    if os.path.exists(path):
        try:
            ds = TablebaseDataset(path)
            check(f"{path}: input_size={expected_input}", ds.input_size == expected_input,
                  f"got {ds.input_size}")
            check(f"{path}: num_pieces={expected_pieces}", ds.num_pieces == expected_pieces,
                  f"got {ds.num_pieces}")
            check(f"{path}: encoding_version={expected_enc}", ds.encoding_version == expected_enc,
                  f"got {ds.encoding_version}")
            check(f"{path}: len > 0", len(ds) > 0)
            x, wdl, dtz = ds[0]
            check(f"{path}: item shape", x.shape[0] == expected_input)
        except Exception as e:
            check(f"{path}: carga sin error", False, str(e))
    else:
        print(f"  SKIP  {path} (no existe)")

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("6. UNRANKING (generate_datasets_parallel)")
print("=" * 60)
from generate_datasets_parallel import (unrank_square_permutation,
                                         unrank_square_combination,
                                         _perm, _choose_coprime_stride)
import math, random

# _perm
check("_perm(64,3) = 64*63*62", _perm(64, 3) == 64*63*62)
check("_perm(64,0) = 1", _perm(64, 0) == 1)
check("_perm(5,6) = 0 (k>n)", _perm(5, 6) == 0)

# unrank_square_permutation: primeros 3 índices deben ser distintos
sq0 = unrank_square_permutation(3, 0)
check("unrank perm idx=0: 3 cuadrados distintos", len(set(sq0)) == 3)
sq1 = unrank_square_permutation(3, 1)
check("unrank perm idx=0 != idx=1", sq0 != sq1)

# unrank es biyectivo en rango pequeño
indices = list(range(100))
perms = [tuple(unrank_square_permutation(3, i)) for i in indices]
check("unrank perm: 100 resultados distintos", len(set(perms)) == 100)

# unrank_square_combination
c0 = unrank_square_combination(3, 0)
check("unrank comb idx=0: 3 cuadrados", len(c0) == 3)

# _choose_coprime_stride
rng = random.Random(42)
stride = _choose_coprime_stride(1000, rng)
check("coprime stride: gcd=1", math.gcd(stride, 1000) == 1)
check("coprime stride: 1 <= stride < 1000", 1 <= stride < 1000)

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("7. METADATA SIDECAR")
print("=" * 60)
import json

meta_files = [
    'data/KPvKP_metadata.json',
    'data/KPvKP_canonical_metadata.json',
    'data/KRRvK_canonical_metadata.json',
    'data/mlp_best_metadata.json',
    'data/mlp_final_metadata.json',
]
for mf in meta_files:
    if os.path.exists(mf):
        try:
            with open(mf, encoding='utf-8') as f:
                meta = json.load(f)
            check(f"{mf}: JSON valido", True)
        except Exception as e:
            check(f"{mf}: JSON valido", False, str(e))
    else:
        print(f"  SKIP  {mf} (no existe)")

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("8. MODEL CHECKPOINT LOADING")
print("=" * 60)

# BUG CONOCIDO en search_poc.py: NeuralSearcher infiere num_pieces desde input_size
# via config_map, pero get_model_for_endgame usa num_pieces para elegir hidden_layers.
# Si se llama con num_pieces=3 para un checkpoint 4-piece (input_size=68),
# se crea un modelo con hidden=[512,512,256,128] en vez de [1024,512,256,128] -> mismatch.
# La solucion correcta es inferir num_pieces desde config_map antes de crear el modelo.
checkpoints = [
    ('data/mlp_best.pth', 'mlp', 4),        # 4-piece KPvKP v4
    ('data/smoke_kqvk_best.pth', 'mlp', 3), # 3-piece KQvK
]
for ckpt_path, model_type, num_pieces in checkpoints:
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            first_key = 'backbone.0.weight'
            if first_key in state:
                input_size = state[first_key].shape[1]
                model = get_model_for_endgame(model_type, num_pieces, input_size=input_size)
                model.load_state_dict(state)
                model.eval()
                x = torch.randn(1, input_size)
                with torch.no_grad():
                    wdl, dtz = model(x)
                check(f"{ckpt_path}: carga y forward OK", wdl.shape[1] in [3, 5])
            else:
                check(f"{ckpt_path}: tiene backbone.0.weight", False, f"keys: {list(state.keys())[:3]}")
        except Exception as e:
            check(f"{ckpt_path}: carga sin error", False, str(e))
    else:
        print(f"  SKIP  {ckpt_path} (no existe)")

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("9. V4 PERSPECTIVE NORMALIZATION")
print("=" * 60)

# Posicion ASIMETRICA con peones: WK a1, WP a4, BK h8, BP h5.
# La posicion anterior (WK/BK en columna e, simetrica) causaba un falso negativo:
# el flip v4 la mapeaba sobre si misma -> encodings identicos para WHITE y BLACK.
# Con una posicion asimetrica se garantiza que el flip produce un encoding distinto.
b_w2 = chess.Board(None)
b_w2.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
b_w2.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.WHITE))
b_w2.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
b_w2.set_piece_at(chess.H5, chess.Piece(chess.PAWN, chess.BLACK))
b_w2.turn = chess.WHITE

b_b2 = b_w2.copy()
b_b2.turn = chess.BLACK

enc_w2 = encode_board(b_w2, relative='v4')
enc_b2 = encode_board(b_b2, relative='v4')

check("v4 WHITE-to-move: 68 dims", enc_w2.shape[0] == 68)
check("v4 BLACK-to-move: 68 dims", enc_b2.shape[0] == 68)
# Con v4, BLACK-to-move flipa el tablero -> encoding diferente al WHITE-to-move
check("v4 BLACK != WHITE encoding (perspectiva diferente)", not np.allclose(enc_w2, enc_b2))

# Pawn progress: orden de piezas en v4 es sorted(color, piece_type)
# chess.PAWN=1, chess.KING=6 -> PAWN < KING en sort
# Para KPvKP WHITE-to-move: WP(0,1), WK(0,6), BP(1,1), BK(1,6)
# WP: dims 0-10, WK: dims 11-21, BP: dims 22-32, BK: dims 33-43
# progress = dim 10 de cada pieza
# Posicion actual: WP en A4 (rank=3), BP en H5 (rank=4)
wp_progress = enc_w2[10]   # WP (pieza 0) progress
bp_progress = enc_w2[32]   # BP (pieza 2) progress
# WP en A4 -> rank 3 -> progress = 3/7
check("v4 pawn progress WP rank4 = 3/7", abs(wp_progress - 3/7) < 1e-5,
      f"got {wp_progress:.4f}, esperado {3/7:.4f}")

# BUG SEMANTICO CORREGIDO: el progreso del peon negro ahora usa (7-rank)/7.0
# porque el peon negro avanza hacia rank 0, no rank 7.
# Para BP en H5 (rank=4), progress correcto = (7-4)/7 = 3/7.
# NOTA: los modelos entrenados con el bug antiguo (rank/7) son incompatibles
# con este encoding corregido y deben reentrenarse.
bp_progress_actual = enc_w2[32]
bp_progress_expected = (7 - chess.square_rank(chess.H5)) / 7.0  # = 3/7
check("v4 BP progress usa (7-rank)/7 (CORREGIDO)", abs(bp_progress_actual - bp_progress_expected) < 1e-5,
      f"got {bp_progress_actual:.4f}, esperado {bp_progress_expected:.4f}")
print(f"  INFO  BP progress={bp_progress_actual:.4f} — bug semántico corregido. Reentrenamiento requerido.")

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("10. AUGMENTATION (horizontal flip en v4)")
print("=" * 60)
from train import TablebaseDataset

smoke_kpvkp = 'data/smoke/KPvKP.npz'
if os.path.exists(smoke_kpvkp):
    ds = TablebaseDataset(smoke_kpvkp)
    x0, _, _ = ds[0]
    # augment_horizontal_flip debe devolver tensor del mismo shape
    x_aug = ds.augment_horizontal_flip(x0)
    check("augment_horizontal_flip: mismo shape", x_aug.shape == x0.shape)
    # Aplicar dos veces debe devolver el original
    x_aug2 = ds.augment_horizontal_flip(x_aug)
    check("augment_horizontal_flip involutivo", torch.allclose(x_aug2, x0, atol=1e-5))
else:
    print(f"  SKIP  {smoke_kpvkp} (no existe)")

# ─────────────────────────────────────────────
print()
print("=" * 60)
print("RESUMEN")
print("=" * 60)
total = PASS + FAIL
print(f"  Pasados: {PASS}/{total}")
print(f"  Fallados: {FAIL}/{total}")
if FAIL == 0:
    print("  TODOS LOS TESTS PASARON")
else:
    print("  HAY FALLOS - revisar arriba")
sys.exit(0 if FAIL == 0 else 1)
