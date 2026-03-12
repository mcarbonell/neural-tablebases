"""Analyze training log without matplotlib"""
import re

def parse_log(log_file):
    """Parse training log and extract metrics"""
    epochs = []
    train_acc = []
    val_acc = []
    val_loss = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Time:' in line:
                epoch_match = re.search(r'Epoch (\d+)/\d+', line)
                if epoch_match:
                    epochs.append(int(epoch_match.group(1)))
                
                train_acc_match = re.search(r'Train Acc: ([\d.]+)', line)
                val_acc_match = re.search(r'Val Acc: ([\d.]+)', line)
                val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
                
                if train_acc_match:
                    train_acc.append(float(train_acc_match.group(1)))
                if val_acc_match:
                    val_acc.append(float(val_acc_match.group(1)))
                if val_loss_match:
                    val_loss.append(float(val_loss_match.group(1)))
    
    return epochs, train_acc, val_acc, val_loss

log_file = 'logs/train_mlp_20260312_214059.log'
epochs, train_acc, val_acc, val_loss = parse_log(log_file)

print("=" * 70)
print("ANÁLISIS DE ENTRENAMIENTO - MLP MEJORADO")
print("=" * 70)

print(f"\n📊 ESTADÍSTICAS GENERALES")
print(f"   Total de épocas: {len(epochs)}")
print(f"   Modelo: MLP con 529,028 parámetros")
print(f"   Dataset: KQvK (368,452 posiciones)")

print(f"\n🎯 MEJORES RESULTADOS")
best_val_acc = max(val_acc)
best_val_acc_epoch = epochs[val_acc.index(best_val_acc)]
best_val_loss = min(val_loss)
best_val_loss_epoch = epochs[val_loss.index(best_val_loss)]

print(f"   Mejor Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) en época {best_val_acc_epoch}")
print(f"   Mejor Val Loss:     {best_val_loss:.4f} en época {best_val_loss_epoch}")

print(f"\n📈 PROGRESO")
print(f"   Época 1:   Val Acc = {val_acc[0]:.4f} ({val_acc[0]*100:.2f}%)")
print(f"   Época 10:  Val Acc = {val_acc[9]:.4f} ({val_acc[9]*100:.2f}%)")
print(f"   Época 50:  Val Acc = {val_acc[49]:.4f} ({val_acc[49]*100:.2f}%)")
print(f"   Época 100: Val Acc = {val_acc[99]:.4f} ({val_acc[99]*100:.2f}%)")
print(f"   Mejora total: {(val_acc[-1] - val_acc[0])*100:.2f} puntos porcentuales")

print(f"\n📉 ESTABILIDAD")
mean_val_acc = sum(val_acc) / len(val_acc)
variance = sum((x - mean_val_acc)**2 for x in val_acc) / len(val_acc)
std_dev = variance ** 0.5

print(f"   Media Val Acc:       {mean_val_acc:.4f} ({mean_val_acc*100:.2f}%)")
print(f"   Desviación estándar: {std_dev:.4f} ({std_dev*100:.2f}%)")
print(f"   Rango: [{min(val_acc):.4f}, {max(val_acc):.4f}]")

# Últimas 20 épocas
last_20_mean = sum(val_acc[-20:]) / 20
last_20_std = (sum((x - last_20_mean)**2 for x in val_acc[-20:]) / 20) ** 0.5

print(f"\n🔍 ÚLTIMAS 20 ÉPOCAS")
print(f"   Media Val Acc:       {last_20_mean:.4f} ({last_20_mean*100:.2f}%)")
print(f"   Desviación estándar: {last_20_std:.4f} ({last_20_std*100:.2f}%)")
print(f"   Tendencia: {'Estancado' if last_20_std > 0.02 else 'Estable'}")

print(f"\n⚠️  DIAGNÓSTICO")
if best_val_acc < 0.65:
    print(f"   ❌ Accuracy baja (<65%): El modelo no está aprendiendo bien")
if std_dev > 0.03:
    print(f"   ⚠️  Alta varianza: El modelo es inestable")
if last_20_std > 0.02:
    print(f"   ⚠️  Oscilación en épocas finales: No converge")
if abs(train_acc[-1] - val_acc[-1]) < 0.01:
    print(f"   ✓ Sin overfitting: Train/Val accuracy similares")

print(f"\n💡 COMPARACIÓN CON BASELINE")
baseline_loss_only = 0.5452  # Predecir siempre "Loss"
baseline_random = 0.3333     # Predicción aleatoria
print(f"   Baseline (siempre Loss): {baseline_loss_only*100:.2f}%")
print(f"   Baseline (aleatorio):    {baseline_random*100:.2f}%")
print(f"   Modelo actual:           {best_val_acc*100:.2f}%")
print(f"   Mejora vs Loss-only:     +{(best_val_acc - baseline_loss_only)*100:.2f}%")
print(f"   Mejora vs aleatorio:     +{(best_val_acc - baseline_random)*100:.2f}%")

print(f"\n🎓 CONCLUSIÓN")
if best_val_acc > 0.90:
    print(f"   ✅ EXCELENTE: El modelo ha aprendido muy bien")
elif best_val_acc > 0.75:
    print(f"   ✓ BUENO: El modelo aprende pero puede mejorar")
elif best_val_acc > 0.60:
    print(f"   ⚠️  LIMITADO: El modelo aprende poco, necesita cambios")
else:
    print(f"   ❌ MALO: El modelo apenas supera el baseline")

print("\n" + "=" * 70)
