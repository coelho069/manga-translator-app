# Performance Report

Data: 2026-05-13

## Gargalos Encontrados

- O detector YOLO salvava imagens de debug em todo processamento, mesmo com debug desligado.
- O OCR era chamado para crops vazios ou quase sem pixels escuros.
- O OCR nao tinha cache por crop, entao regioes duplicadas podiam ser lidas novamente.
- A limpeza detectava pixels escuros na imagem inteira para cada balao.
- A renderizacao testava muitas fontes e recalculava quebras de linha sem necessidade.
- O modo de performance existia na interface, mas nao ajustava `yolo_imgsz`.

## Alteracoes Feitas

- `fast`, `balanced` e `quality` agora ajustam `yolo_imgsz`:
  - `fast`: 640
  - `balanced`: 960
  - `quality`: 1280
- Debug de deteccao so e salvo quando `debug_enabled=True` ou `debug_dir` existe.
- OCR ganhou cache por hash do crop, separado por idioma e uso de GPU.
- OCR agora pula crops pequenos ou sem sinal minimo de texto escuro.
- Tradutor reutiliza o cliente do `deep-translator` por par de idiomas e deduplica frases por lote.
- Pipeline deixou de salvar `bubble_flow_report.json` no modo normal; salva apenas quando debug esta ativo.
- Limpeza por pixels escuros passou a trabalhar no ROI do balao em vez da imagem inteira.
- Modo `fast` reduz raios morfologicos de limpeza e limita tentativas de fonte.
- Renderizador reutiliza quebra de linha por tamanho de fonte e respeita limite de tentativas.
- Timings por etapa continuam em `result.metadata["timings"]`.

## Tempos Antes/Depois

Nao foi executado benchmark completo com OCR/traducao nesta rodada para evitar chamadas pesadas e dependentes de rede. A instrumentacao atual permite comparar diretamente no app pelos campos:

```text
result.metadata["timings"]["detect"]
result.metadata["timings"]["ocr"]
result.metadata["timings"]["translate"]
result.metadata["timings"]["clean"]
result.metadata["timings"]["render"]
result.metadata["timings"]["save"]
result.metadata["timings"]["total"]
```

## Ganhos Esperados

- Menos tempo de IO por remover debug automatico no fluxo normal.
- Menos chamadas ao PaddleOCR em baloes vazios, pequenos ou duplicados.
- Menos chamadas externas de traducao em textos repetidos.
- Limpeza mais rapida em paginas grandes por processar apenas o ROI de cada balao.
- Renderizacao mais rapida no modo `fast` por reduzir tentativas de fonte.

## Recomendacoes Futuras

- Criar um pequeno conjunto fixo de imagens de benchmark em `examples/bubbles/`.
- Registrar automaticamente comparativos por modo: `fast`, `balanced`, `quality`.
- Avaliar OCR paralelo somente se PaddleOCR se mostrar seguro no ambiente Windows/CPU.
- Considerar um tradutor local ou API com batch real para reduzir latencia de rede.
