# Manga Translator App

Aplicativo web local para Windows que recebe uma página de mangá em inglês, detecta balões de fala, lê o texto com OCR, traduz para português, apaga o texto original e escreve a tradução dentro dos balões.

O aplicativo usa Streamlit e roda no navegador local. O processamento padrão é em CPU.

## Estrutura

```text
manga-translator-app/
  app/
    config.py
    types.py
    utils.py
    detector.py
    ocr.py
    translator.py
    cleaner.py
    renderer.py
    pipeline.py
    backend.py
  models/
  output/
  examples/
  streamlit_app.py
```

## Instalação no Windows

Requisitos:

- Windows
- Python 3.10 instalado
- PowerShell

Na pasta do projeto, execute:

```powershell
.\setup.ps1
```

O script cria `.venv`, instala PyTorch CPU, instala as dependências e cria as pastas necessárias.

## Modelo YOLO

Coloque o modelo de segmentação de balões neste caminho:

```text
models/bubble_seg.pt
```

O modelo esperado deve ter a classe:

```python
{0: "speech bubble"}
```

Sem esse arquivo, o aplicativo mostra uma mensagem clara informando que o modelo não foi encontrado.

## Como rodar

Depois da instalação:

```powershell
.\run_app.ps1
```

Ou manualmente:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

O Streamlit abrirá o endereço local no navegador.

## Como usar

1. Envie uma imagem PNG, JPG ou JPEG.
2. Confira a pré-visualização.
3. Clique em **Iniciar tradução**.
4. Aguarde as etapas de progresso.
5. Compare a imagem original e a traduzida lado a lado.
6. Confira a lista de textos detectados.
7. Baixe a imagem traduzida.

As imagens processadas ficam em `output/`, separadas por diretório de trabalho.

## Criar imagem de teste

Para gerar uma imagem simples de exemplo:

```powershell
.\.venv\Scripts\Activate.ps1
python examples\create_sample.py
```

Isso cria `examples/sample_manga_page.png`.

## Criar exemplos de balões

Para gerar um conjunto de imagens PNG simples com balões de fala:

```powershell
.\.venv\Scripts\Activate.ps1
python examples\create_bubble_examples.py
```

As imagens serão criadas em:

```text
examples/bubbles/
```

Arquivos gerados:

- `simple_bubble_01.png`
- `simple_bubble_02.png`
- `simple_bubble_03.png`
- `long_text_bubble.png`
- `small_bubble.png`
- `double_bubble.png`
- `noisy_bubble.png`

No Streamlit, marque **Usar imagem de exemplo** para carregar uma dessas imagens sem fazer upload manual. O upload normal continua funcionando.

## Melhorando a detecção dos balões

A pasta `dataset/bubbles/` foi criada para organizar imagens anotadas e treinar/fine-tunar um modelo YOLO segmentation melhor para balões de fala de mangá.

Importante: colocar imagens na pasta não melhora automaticamente o app. É preciso:

1. Colocar imagens em `dataset/bubbles/images/train` e `dataset/bubbles/images/val`.
2. Anotar os balões como segmentação/polígono YOLO, não apenas bbox.
3. Salvar os labels correspondentes em `dataset/bubbles/labels/train` e `dataset/bubbles/labels/val`.
4. Treinar/fine-tunar o modelo.
5. Copiar o melhor peso treinado para `models/bubble_seg.pt`.

Exemplo de pares imagem/label:

```text
dataset/bubbles/images/train/page_001.png
dataset/bubbles/labels/train/page_001.txt
```

Classe única:

```text
0 = speech bubble
```

Para treinar em CPU:

```powershell
python training/train_bubble_model.py --epochs 50 --imgsz 640 --batch 4 --device cpu
```

Treino em CPU pode ser lento. Com GPU compatível, use algo como:

```powershell
python training/train_bubble_model.py --epochs 50 --imgsz 640 --batch 4 --device 0
```

Depois do treino, o melhor modelo geralmente fica em:

```text
runs/segment/train/weights/best.pt
```

Para usar esse modelo no app, copie o arquivo para:

```text
models/bubble_seg.pt
```

Para gerar imagens de debug com máscaras/bboxes desenhados:

```powershell
python training/predict_debug.py --model models/bubble_seg.pt --source dataset/bubbles/images/val
```

As predições serão salvas em:

```text
output/debug_predictions/
```

## Arquitetura

- `detector.py`: carrega o YOLO e segmenta os balões.
- `ocr.py`: usa PaddleOCR em inglês e converte polígonos locais para coordenadas globais.
- `translator.py`: traduz com `deep-translator`, com cache e fallback para o texto original.
- `cleaner.py`: cria máscara do texto, restringe à área interna do balão e aplica `cv2.inpaint`.
- `renderer.py`: calcula área segura, quebra linhas e centraliza a tradução com PIL.
- `pipeline.py`: coordena todas as etapas e salva a imagem final.
- `streamlit_app.py`: interface local no navegador.

## Problemas comuns

### Modelo não encontrado

Verifique se o arquivo existe exatamente em:

```text
models/bubble_seg.pt
```

### OCR lento na primeira execução

O PaddleOCR pode baixar ou carregar modelos na primeira vez. Depois tende a ficar mais rápido.

### Tradução falhou

O tradutor usa serviço online via `deep-translator`. Se a rede falhar, o app mantém o texto original em vez de quebrar.

### Nenhum balão detectado

O app não quebra. Ele salva a imagem original como resultado traduzido e mostra o aviso na interface. Confira se o modelo YOLO é adequado para balões de mangá.

## Comandos de teste

Verificar sintaxe dos arquivos:

```powershell
python -m compileall app streamlit_app.py examples
```

Gerar imagem de exemplo:

```powershell
python examples\create_sample.py
```

Rodar o app:

```powershell
python -m streamlit run streamlit_app.py
```
