# Treino do modelo de balões

Esta pasta contém scripts para treinar/fine-tunar e inspecionar um modelo YOLO segmentation para detectar balões de fala.

## Treinar

Antes de treinar, coloque imagens e labels em:

```text
dataset/bubbles/
```

Depois execute:

```powershell
python training/train_bubble_model.py --epochs 50 --imgsz 640 --batch 4 --device cpu
```

Por padrão, o script usa:

```text
models/bubble_seg.pt
```

como modelo inicial, e:

```text
dataset/bubbles/data.yaml
```

como dataset.

Treino em CPU pode ser lento. Se tiver GPU compatível, use `--device 0`.

## Gerar predições de debug

Para ver como o modelo está detectando os balões em imagens de validação:

```powershell
python training/predict_debug.py --model models/bubble_seg.pt --source dataset/bubbles/images/val
```

As imagens anotadas serão salvas em:

```text
output/debug_predictions/
```

