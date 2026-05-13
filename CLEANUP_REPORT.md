# Relatorio de Limpeza

Data: 2026-05-13

## Objetivo

Limpar o projeto `manga-translator-app`, mantendo apenas os arquivos usados pelo aplicativo Streamlit atual e removendo artefatos de treino, debug, cache e saidas antigas.

## Arquivos Removidos

- `examples/create_sample.py`
  - Script antigo de exemplo que nao era usado pelo fluxo atual do app.
- `examples/sample_manga_page.png`
  - Imagem gerada por script antigo e nao necessaria para execucao.

## Pastas Removidas

- `dataset/`
  - Estrutura de dataset para treino futuro. Nao era importada pelo app principal.
- `training/`
  - Scripts de treino e predicao manual. Nao faziam parte do fluxo runtime do Streamlit.
- `runs/`
  - Resultado antigo de treino YOLO. Artefato gerado e nao necessario para execucao.
- Conteudo antigo de `output/`
  - Foram removidos jobs antigos, imagens traduzidas, debug visual e relatorios gerados.
  - Foi mantido apenas `output/.gitkeep`.
- `__pycache__/`
  - Caches Python criados durante validacao foram removidos apos o teste.

## Arquivos e Pastas Mantidos

- `streamlit_app.py`
  - Interface principal do aplicativo.
- `app/`
  - Codigo runtime do backend, pipeline, detector, OCR, tradutor, limpeza e renderizacao.
- `models/bubble_seg.pt`
  - Modelo YOLO segmentation atualmente usado pelo detector.
- `models/.gitkeep`
  - Mantem a pasta `models/` no repositorio mesmo sem publicar pesos `.pt`.
- `output/.gitkeep`
  - Mantem a pasta `output/`; o app recria os resultados em tempo de execucao.
- `examples/bubbles/`
  - Imagens de exemplo ainda usadas opcionalmente pela interface Streamlit.
- `examples/create_bubble_examples.py`
  - Script ainda referenciado pela interface para gerar exemplos locais.
- `requirements.txt`
  - Lista enxuta de dependencias runtime.
- `README.md`
  - Atualizado para refletir a estrutura limpa.
- `setup.ps1` e `run_app.ps1`
  - Scripts de instalacao e execucao no Windows.
- `.venv/`
  - Mantida localmente por seguranca para validacao; continua ignorada pelo Git.

## Dependencias Removidas

- `torchaudio==2.2.2`
  - Nao era importada pelo app. O projeto usa PyTorch/YOLO em imagem, mas nao audio.
- `tqdm>=4.66.0`
  - Nao era importada diretamente pelo app runtime.

## Dependencias Mantidas

- `opencv-python`
  - Manipulacao de imagem, mascaras e inpainting.
- `numpy`
  - Operacoes numericas e mascaras.
- `pillow`
  - Renderizacao de texto e geracao de exemplos.
- `torch` e `torchvision`
  - Necessarias para execucao do YOLO/Ultralytics em CPU.
- `ultralytics`
  - Detector YOLO segmentation.
- `paddleocr` e `paddlepaddle`
  - OCR dos baloes.
- `deep-translator`
  - Traducao para portugues.
- `huggingface_hub`
  - Download/cache automatico do modelo Hugging Face.
- `streamlit`
  - Interface web local.

## Arquivos Atualizados

- `.gitignore`
  - Passou a manter `output/.gitkeep` enquanto ignora saidas geradas.
  - Inclui `runs/`, `*.log`, `*.tmp`, `.pytest_cache/`, `__pycache__/`, `.venv/` e `manga109-segmentation-bubble/`.
- `requirements.txt`
  - Removidas dependencias nao usadas diretamente.
- `setup.ps1`
  - Instalacao do PyTorch CPU ajustada sem `torchaudio`.
- `README.md`
  - Removidas referencias a dataset, treino, `create_sample.py` e scripts apagados.

## Comandos Usados Para Validar

```powershell
rg --files
rg -n "dataset|training|create_sample|sample_manga_page|train_bubble_model|predict_debug|debug_predictions|runs/segment|torchaudio|tqdm" README.md requirements.txt setup.ps1 streamlit_app.py app
python -m compileall app streamlit_app.py examples
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
python -c "from ultralytics import YOLO; m=YOLO('models/bubble_seg.pt'); print(m.names)"
```

## Resultado da Validacao

- `compileall` concluiu sem erros.
- O Streamlit iniciou em modo headless e respondeu HTTP 200 em `http://localhost:8502`.
- O modelo YOLO principal carregou corretamente e retornou `{0: 'speech bubble'}`.
- A pasta `output/` ficou limpa, contendo apenas `.gitkeep`.
- O modelo `models/bubble_seg.pt` foi preservado.

## Observacoes

- A limpeza nao removeu a pasta `.venv/` porque ela pode ser util para execucao local e validacao, mas ela permanece ignorada pelo Git.
- As imagens de `examples/bubbles/` foram mantidas porque a interface atual permite selecionar exemplos dessa pasta.
- O fluxo principal do app nao foi alterado nesta limpeza.
