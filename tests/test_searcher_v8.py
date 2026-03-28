import unittest
import chess
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search.searcher_v8 import GnnSearcher

class TestGnnSearcher(unittest.TestCase):
    @patch('search.searcher_v8.ChessGnnV8_Pro')
    @patch('search.searcher_v8.torch.load')
    @patch('search.searcher_v8.RustGnnEngine')
    @patch('search.searcher_v8.chess.syzygy.open_tablebase')
    def setUp(self, mock_tb, mock_engine, mock_load, mock_model):
        self.mock_model_instance = MagicMock()
        mock_model.return_value = self.mock_model_instance
        self.searcher = GnnSearcher("dummy_model.pth", "dummy_syzygy_path", device="cpu")
        
    def test_invert_score(self):
        self.assertEqual(self.searcher._invert_score(0.0), 2.0)
        self.assertEqual(self.searcher._invert_score(2.0), 0.0)
        self.assertEqual(self.searcher._invert_score(1.0), 1.0)
        self.assertEqual(self.searcher._invert_score(0.5), 1.5)
        
    def test_evaluate_game_over_win(self):
        # Scholar's mate: White wins
        board = chess.Board("r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        score = self.searcher.evaluate(board)
        self.assertEqual(score, 2.0)
        
    def test_evaluate_game_over_draw(self):
        # Stalemate => Draw
        board = chess.Board("8/8/8/8/8/2k5/2q5/K7 w - - 0 1")
        score = self.searcher.evaluate(board)
        self.assertEqual(score, 1.0)

    @patch.object(GnnSearcher, '_gnn_eval')
    def test_minimax_depth_0(self, mock_eval):
        # Depth 0 should just return evaluate()
        mock_eval.return_value = 1.5
        # King + Pawn vs King (not game over)
        board = chess.Board("8/8/8/8/4K3/4P3/8/k7 w - - 0 1")
        score = self.searcher.minimax(board, depth=0, alpha=-float('inf'), beta=float('inf'), is_maximizing=True)
        self.assertEqual(score, 1.5)
        mock_eval.assert_called_once_with(board)

    @patch.object(GnnSearcher, 'evaluate')
    def test_minimax_mate_in_1(self, mock_evaluate):
        # A simple mate in 1. End of an underpromotion or something.
        # Actually, let's just make the side-effect evaluate return 2.0 if White wins
        
        # Here is a mate in 1: White Queen on a7, Black King on a8, White King on c8.
        # White can play Qb7#
        board = chess.Board("k1K5/Q7/8/8/8/8/8/8 w - - 0 1")
        
        def eval_side_effect(b):
            if b.is_game_over():
                result = b.result()
                if result == "1-0":
                    return 2.0
                elif result == "0-1":
                    return 0.0
                return 1.0
            return 1.0 # default
            
        mock_evaluate.side_effect = eval_side_effect
        
        score = self.searcher.minimax(board, depth=1, alpha=-float('inf'), beta=float('inf'), is_maximizing=True)
        self.assertEqual(score, 2.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
