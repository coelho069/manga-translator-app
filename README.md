# Manga Translator App

Aplicativo Python com interface Streamlit para traduzir páginas de mangá a partir de imagens ou PDFs.

O sistema recebe uma página, detecta balões de fala, executa OCR no recorte de cada balão, traduz o texto, apaga o texto original dentro da máscara do balão e renderiza a tradução na imagem final.

Fluxo principal:

```text
upload -> detecção de balões -> OCR -> tradução -> limpeza do texto original -> renderização -> resultado final
```

## O Que O Projeto Faz

- Recebe imagens `PNG`, `JPG`, `JPEG` ou arquivos `PDF`.
- Converte páginas de PDF em imagens com PyMuPDF.
- Detecta e segmenta balões de fala com YOLOv8 segmentation.
- Faz OCR do texto dentro dos balões pelo módulo `app/ocr.py`.
- Traduz o texto com CTranslate2 usando modelo SMALL100 baixado do Hugging Face.
- Remove o texto original dentro da máscara de cada balão.
- Renderiza a tradução dentro da área segura do balão.
- Mostra o resultado na interface Streamlit e permite baixar as páginas traduzidas.
- Salva os resultados em `output/`.

Observação: o código atual usa `PaddleOCR` para OCR. Não há dependência `manga-ocr` no `requirements.txt`.

## Tecnologias Usadas

- Python 3.10.
- Streamlit para a interface web.
- Ultralytics YOLOv8 segmentation para detectar balões.
- Modelo de balões `kitsumed/yolov8m_seg-speech-bubble` via Hugging Face.
- PaddleOCR/PaddlePaddle para OCR.
- CTranslate2 e SentencePiece para tradução.
- Modelo `entai2965/small100-ctranslate2` para tradução.
- Pillow e OpenCV para manipulação, limpeza e renderização de imagem.
- PyMuPDF para converter PDFs em páginas PNG.

Não há API FastAPI nem serviço systemd versionado neste repositório.

## Estrutura Do Projeto

```text
manga-translator-app/
  app/
    backend.py        # Entrada usada pela UI; cria jobs e chama o pipeline
    cleaner.py        # Limpeza/inpainting/fill do texto original nos balões
    config.py         # Configurações padrão do pipeline
    detector.py       # Carregamento do YOLO e segmentação dos balões
    ocr.py            # OCR dos crops de balão
    pdf_utils.py      # Conversão de PDF para imagens
    pipeline.py       # Orquestra detecção, OCR, tradução, limpeza e renderização
    renderer.py       # Desenha a tradução dentro do balão
    translator.py     # Tradução com CTranslate2/SMALL100
    types.py          # Tipos compartilhados do pipeline
    utils.py          # Utilidades gerais e modos de tradução
  examples/           # Imagens/scripts de exemplo
  models/             # Modelo YOLO local, quando baixado ou copiado
  output/             # Jobs, imagens traduzidas e arquivos de debug
  streamlit_app.py    # Interface principal Streamlit
  requirements.txt    # Dependências Python
  packages.txt        # Pacotes de sistema úteis em Linux/VPS
  setup.ps1           # Instalação local no Windows
  run_app.ps1         # Execução local no Windows
```

## Modos De Tradução

Os modos disponíveis ficam em `app/utils.py`:

```text
en_to_pt  -> Inglês para Português
ja_to_en  -> Japonês para Inglês
```

O destino padrão do `AppConfig` é configurável no código. A interface usa os modos definidos em `TRANSLATION_MODES`.

## Modelo De Detecção De Balões

O detector espera o modelo em:

```text
models/bubble_seg.pt
```

Se o arquivo não existir e `auto_download_bubble_model=True`, o código baixa automaticamente:

```text
repo: kitsumed/yolov8m_seg-speech-bubble
arquivo: model.pt
destino local: models/bubble_seg.pt
```

O YOLO é usado para obter, por balão:

- `bbox`;
- máscara segmentada;
- confiança;
- crop seguro para OCR;
- área segura para limpeza e renderização.

## Instalação Local

### Windows

Na raiz do projeto:

```powershell
.\setup.ps1
```

Ou manualmente:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Para rodar:

```powershell
.\run_app.ps1
```

Ou:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

### Linux/VPS

Instale os pacotes de sistema listados em `packages.txt` ou equivalentes da distribuição:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
xargs -a packages.txt sudo apt install -y
```

Crie o ambiente e instale as dependências:

```bash
cd /caminho/para/manga-translator-app
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Rode a aplicação:

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Acesse no navegador:

```text
http://SEU_SERVIDOR:8501
```

## Como Usar

1. Abra a interface Streamlit.
2. Escolha o modo de tradução.
3. Ajuste performance e fonte na barra lateral, se necessário.
4. Envie uma ou mais imagens, ou um PDF.
5. Aguarde o processamento página por página.
6. Baixe cada página traduzida ou o `.zip` final.

Os arquivos gerados ficam em subpastas de:

```text
output/
```

## Como Rodar Na VPS

Execução direta:

```bash
cd /caminho/para/manga-translator-app
source .venv/bin/activate
python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Para manter rodando em uma sessão simples:

```bash
cd /caminho/para/manga-translator-app
source .venv/bin/activate
nohup python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 > output/streamlit.log 2>&1 &
```

Ver logs nesse modo:

```bash
tail -f output/streamlit.log
```

Este repositório não contém arquivo `.service` do systemd. Se a VPS tiver um serviço criado manualmente, os comandos normalmente serão:

```bash
sudo systemctl status manga-translator
sudo journalctl -u manga-translator -f
sudo systemctl restart manga-translator
```

O nome do serviço depende da configuração feita no servidor.

## Variáveis De Ambiente

Não há arquivo `.env` nem leitura direta de variáveis de ambiente no código atual.

As principais configurações ficam em `app/config.py` e em controles da interface Streamlit:

```text
translation_model
translation_cache_enabled
translation_timeout_seconds
translation_beam_size
source_lang
target_lang
ocr_lang
yolo_confidence
yolo_iou
yolo_imgsz
use_gpu
debug_enabled
```

Se o projeto for adaptado para `.env` em deploy, não salve credenciais no Git.

## Debug E Saídas

Quando o debug está ativado, o pipeline pode salvar imagens intermediárias e relatórios em `output/`, incluindo máscaras, caixas detectadas e imagens antes/depois da limpeza.

Arquivos comuns:

```text
output/debug_detection.png
output/debug_masks.png
output/debug_boxes.json
output/job_.../translated_*.png
```

## Validação Rápida

Verificar sintaxe dos módulos:

```bash
python -m compileall app streamlit_app.py
```

Verificar se as dependências principais importam:

```bash
python - <<'PY'
import cv2
import numpy
import PIL
import streamlit
import ultralytics
import huggingface_hub
print("Dependências principais OK")
PY
```

Rodar a aplicação localmente:

```bash
python -m streamlit run streamlit_app.py
```

## Observações

- A primeira execução pode demorar porque modelos do Hugging Face e PaddleOCR podem ser baixados para cache local.
- O processamento padrão usa CPU.
- Modelos e outputs podem ocupar bastante espaço em disco.
- A qualidade final depende da segmentação dos balões, do OCR e do modelo de tradução.
