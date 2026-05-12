# Dataset de balões de fala

Esta pasta guarda imagens e anotações para treinar ou ajustar um modelo YOLO segmentation especializado em balões de fala de mangá.

## Estrutura

```text
dataset/bubbles/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

## Onde colocar os arquivos

- `images/train`: imagens usadas no treino.
- `images/val`: imagens usadas na validação.
- `labels/train`: labels YOLO segmentation das imagens de treino.
- `labels/val`: labels YOLO segmentation das imagens de validação.

Cada imagem precisa ter um arquivo `.txt` com o mesmo nome na pasta de labels correspondente.

Exemplo:

```text
images/train/page_001.png
labels/train/page_001.txt
```

Para validação:

```text
images/val/page_020.png
labels/val/page_020.txt
```

## Formato das anotações

As labels devem ser de segmentação, com polígonos, não apenas bounding boxes.

Formato YOLO segmentation:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

As coordenadas devem ser normalizadas entre `0` e `1`.

Classe única:

```text
0 = speech bubble
```

Exemplo simplificado de uma linha de label:

```text
0 0.312 0.210 0.438 0.198 0.520 0.285 0.498 0.410 0.350 0.430
```

Use uma ferramenta de anotação compatível com YOLO segmentation para desenhar os contornos dos balões.

