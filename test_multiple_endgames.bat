@echo off
echo ========================================
echo Testing Multiple Endgames
echo ========================================

echo.
echo Generating KPvK dataset...
python src/generate_datasets.py --config KPvK --relative

echo.
echo Training KPvK...
python src/train.py --data_path data/KPvK.npz --model mlp --epochs 30 --batch_size 2048 --lr 0.001 --patience 20

echo.
echo ========================================
echo All tests complete!
echo ========================================
pause
