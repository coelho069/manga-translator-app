# Manga Translator App

Aplicativo local em Python com Streamlit para traduzir páginas de mangá em inglês ou chinês simplificado para português.  
O app recebe uma imagem, detecta balões de fala com YOLO segmentation, lê o texto com PaddleOCR, traduz com `deep-translator`, tenta apagar o texto original e renderiza a tradução dentro dos balões.

O processamento roda localmente no Windows e usa CPU por padrão.

## Funcionalidades

- Upload de imagens PNG, JPG e JPEG.
- Pré-visualização da imagem original.
- Detecção de balões de fala com YOLO segmentation.
- OCR em inglês ou chinês simplificado com PaddleOCR.
- Tradução automática para português com `deep-translator`.
- Limpeza do texto original dentro dos balões.
- Renderização da tradução centralizada no balão.
- Comparação lado a lado entre imagem original e traduzida.
- Botão para baixar a imagem traduzida.
- Execução local no navegador com Streamlit.
- Geração de imagens de exemplo para testes.
- Estrutura opcional de dataset e scripts de treino para melhorar o modelo de balões.

## Idiomas Suportados

O app aceita dois idiomas de entrada nesta versão:

| Entrada | OCR PaddleOCR | Tradução deep-translator | Saída |
| --- | --- | --- | --- |
| Inglês | `en` | `en -> pt` | Português |
| Chinês simplificado | `ch` | `zh-CN -> pt` | Português |

Na interface Streamlit, selecione **Idioma de entrada** antes de iniciar a tradução. Se um valor inválido for recebido internamente, o app usa inglês como fallback.

## Qualidade da Tradução

O app possui dois estilos de tradução:

- **Natural**: prioriza português brasileiro mais natural, com pequenos ajustes para expressões comuns de mangá/comic.
- **Literal**: usa uma tradução mais direta, com menos ajustes idiomáticos.

No modo **Natural**, o tradutor aplica algumas melhorias antes e depois da tradução:

- tradução em lote por página, mantendo a ordem dos balões;
- cache para frases repetidas, ajudando na consistência;
- glossário simples para preservar honoríficos como `senpai`, `sensei`, `kun`, `chan` e `san`;
- proteção básica de possíveis nomes próprios antes de enviar o texto ao tradutor;
- normalização de pontuação em português, preservando `!`, `?`, `...` e combinações emocionais.

Exemplos de expressões tratadas com mais naturalidade:

```text
No way! -> Não pode ser!
I got it! -> Entendi!
Shut up! -> Cala a boca!
Damn it! -> Droga!
```

A tradução continua sendo automática e pode falhar em frases ambíguas, piadas, nomes incomuns ou contexto visual que não esteja no texto OCR.

## Performance

O app possui três modos de performance:

- **Equilibrado**: padrão recomendado, mantém boa qualidade com custo moderado.
- **Rápido**: pula balões pequenos com mais agressividade, reduz morfologia e limita tentativas de renderização.
- **Qualidade**: preserva ajustes mais cuidadosos e tende a processar mais balões, com maior custo.

O pipeline registra tempos por etapa em `result.metadata["timings"]`, incluindo:

```text
load_image
detect
ocr
translate
clean
render
save
total
```

Na interface, o resultado mostra o tempo total e permite abrir os tempos por etapa. O debug visual fica desligado por padrão para evitar escrita desnecessária de imagens em disco; habilite **Salvar debug visual** apenas quando estiver investigando problemas.

## Demonstração / Fluxo

1. Instale as dependências do projeto.
2. Coloque o modelo YOLO em `models/bubble_seg.pt`.
3. Rode o aplicativo Streamlit.
4. Escolha o idioma de entrada: inglês ou chinês.
5. Envie uma página de mangá no idioma escolhido.
6. Clique em **Iniciar tradução**.
7. Aguarde as etapas de detecção, OCR, tradução, limpeza e renderização.
8. Baixe a imagem traduzida ou acesse o resultado em `output/`.

## Estrutura do Projeto

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
  dataset/
    bubbles/
  examples/
    bubbles/
    create_bubble_examples.py
    create_sample.py
  models/
    bubble_seg.pt
  output/
  training/
    train_bubble_model.py
    predict_debug.py
  streamlit_app.py
  requirements.txt
  setup.ps1
  run_app.ps1
```

### Arquivos principais

- `app/detector.py`: carrega o modelo YOLO segmentation e detecta os balões de fala.
- `app/ocr.py`: executa OCR em inglês com PaddleOCR e retorna textos/polígonos detectados.
- `app/translator.py`: traduz o texto de inglês para português usando `deep-translator`.
- `app/cleaner.py`: tenta remover o texto original dentro dos balões, preservando a borda.
- `app/renderer.py`: quebra linhas, ajusta fonte e desenha a tradução dentro do balão.
- `app/pipeline.py`: coordena o fluxo completo de detecção, OCR, tradução, limpeza e renderização.
- `app/backend.py`: recebe a imagem enviada, cria diretório de saída e chama o pipeline.
- `streamlit_app.py`: interface web local do aplicativo.

## Requisitos

- Windows.
- Python 3.10.
- PowerShell.
- Ambiente virtual recomendado.
- Processamento padrão em CPU.

> Observação: algumas dependências, especialmente `torch`, `paddlepaddle`, `paddleocr` e `ultralytics`, podem ser pesadas. A instalação inicial pode demorar.

## Instalação

### A) Instalação rápida com script

Na pasta raiz do projeto, execute:

```powershell
.\setup.ps1
```

O script cria o ambiente virtual `.venv`, instala o PyTorch CPU, instala as dependências do `requirements.txt` e cria as pastas necessárias.

### B) Instalação manual

Na pasta raiz do projeto:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Modelo YOLO

O aplicativo precisa de um modelo YOLO segmentation neste caminho:

```text
models/bubble_seg.pt
```

Esse arquivo não vai para o Git por padrão, pois modelos `.pt` costumam ser grandes e podem variar conforme o treino.

O modelo esperado deve detectar a classe:

```text
speech bubble
```

Para verificar as classes do modelo:

```powershell
python -c "from ultralytics import YOLO; m=YOLO('models/bubble_seg.pt'); print(m.names)"
```

O resultado esperado deve conter algo como:

```python
{0: 'speech bubble'}
```

## Como Executar

Com o ambiente configurado:

```powershell
.\run_app.ps1
```

Ou manualmente:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

O Streamlit abrirá o app no navegador local, normalmente em:

```text
http://localhost:8501
```

## Como Usar

1. Abra o app no navegador.
2. Escolha o idioma de entrada: **Inglês** ou **Chinês**.
3. Escolha o estilo de tradução: **Natural** ou **Literal**.
4. Escolha o modo de performance: **Equilibrado**, **Rápido** ou **Qualidade**.
5. Envie uma imagem PNG, JPG ou JPEG.
6. Opcionalmente, marque **Usar imagem de exemplo** para testar com arquivos de `examples/bubbles/`.
7. Clique em **Iniciar tradução**.
8. Acompanhe a barra de progresso.
9. Compare a imagem original e a imagem traduzida lado a lado.
10. Baixe a imagem traduzida pelo botão da interface.
11. Também é possível encontrar os resultados em `output/`.

## Linha de Comando / Testes

Gerar uma página simples de teste:

```powershell
python examples\create_sample.py
```

Gerar exemplos variados de balões:

```powershell
python examples\create_bubble_examples.py
```

Verificar sintaxe dos principais arquivos Python:

```powershell
python -m compileall app streamlit_app.py examples
```

Rodar o app:

```powershell
python -m streamlit run streamlit_app.py
```

## Melhorando a Detecção dos Balões

A pasta `dataset/bubbles/` foi criada para organizar imagens anotadas e treinar ou fine-tunar um modelo YOLO segmentation melhor para balões de fala de mangá.

Importante: colocar imagens na pasta não melhora automaticamente o app. É preciso anotar os balões, treinar o modelo e depois substituir o peso usado pelo aplicativo.

Estrutura do dataset:

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

Cada imagem precisa ter um `.txt` com o mesmo nome na pasta de labels correspondente.

Exemplo:

```text
dataset/bubbles/images/train/page_001.png
dataset/bubbles/labels/train/page_001.txt
```

As labels devem estar no formato YOLO segmentation, com polígonos, não apenas bounding boxes.

Classe única:

```text
0 = speech bubble
```

Treinar em CPU:

```powershell
python training/train_bubble_model.py --epochs 50 --imgsz 640 --batch 4 --device cpu
```

Treino em CPU pode ser lento. Com GPU compatível, é possível usar:

```powershell
python training/train_bubble_model.py --epochs 50 --imgsz 640 --batch 4 --device 0
```

Depois do treino, o melhor modelo geralmente fica em:

```text
runs/segment/train/weights/best.pt
```

Para usar o novo modelo no app, copie esse arquivo para:

```text
models/bubble_seg.pt
```

Gerar predições de debug:

```powershell
python training/predict_debug.py --model models/bubble_seg.pt --source dataset/bubbles/images/val
```

As imagens de debug serão salvas em:

```text
output/debug_predictions/
```

## Problemas Comuns

### Modelo não encontrado

Verifique se o arquivo existe exatamente em:

```text
models/bubble_seg.pt
```

Sem esse arquivo, o detector YOLO não consegue iniciar.

### Erro do Torch `shm.dll` no Windows

Esse erro costuma estar relacionado à instalação do PyTorch ou a conflitos no ambiente Python. Tente recriar a `.venv` e instalar o PyTorch CPU pelo índice oficial:

```powershell
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

### `paddlepaddle` não compatível com Python 3.14

Use Python 3.10. O projeto foi pensado para Python 3.10 no Windows. Versões muito novas do Python podem não ter wheels compatíveis para `paddlepaddle`.

### App escrevendo a tradução por cima do texto original

Isso indica que a etapa de limpeza não conseguiu gerar uma máscara boa para remover o texto antigo. Verifique os arquivos de debug em `output/<job>/debug/`, especialmente:

```text
debug_dark_text_mask.png
debug_ocr_mask.png
debug_final_cleanup_mask.png
debug_after_cleanup.png
```

Se a máscara final não cobre o texto antigo, ajuste as configurações de limpeza em `app/config.py` ou melhore a segmentação dos balões.

### Nenhum balão detectado

Possíveis causas:

- O modelo `bubble_seg.pt` não foi treinado para esse estilo de mangá.
- A imagem está em baixa resolução.
- O balão tem formato incomum.
- A confiança do YOLO está alta demais.

Confira o modelo e, se necessário, gere predições com:

```powershell
python training/predict_debug.py --model models/bubble_seg.pt --source caminho\da\imagem.png
```

### Tradução falhando por internet

O `deep-translator` depende de acesso externo ao serviço de tradução. Se a rede falhar, o app tenta manter o texto original em vez de quebrar o fluxo.

### OCR lento na primeira execução

O PaddleOCR pode demorar na primeira execução por carregamento inicial dos modelos. Depois disso, a execução tende a ser mais rápida.

## Limitações Atuais

- Páginas reais de mangá variam muito em qualidade, resolução, contraste e estilo visual.
- A qualidade da detecção depende diretamente do modelo `models/bubble_seg.pt`.
- O OCR pode falhar em fontes muito estilizadas, texto inclinado, ruído ou baixa resolução.
- A limpeza do texto original ainda pode falhar em alguns balões, especialmente quando há arte ou sombras dentro do balão.
- A tradução automática pode não preservar contexto, tom, gênero ou estilo de fala com perfeição, mesmo com o modo Natural.
- O app foi pensado para uso local e processamento de imagens individuais, não para produção em lote em larga escala.

## Melhorias Futuras

- Treinar um modelo próprio de segmentação com mais exemplos de mangá.
- Melhorar o debug visual dos balões e máscaras dentro da interface.
- Implementar técnicas de inpainting mais avançadas.
- Suporte a mais idiomas de origem e destino.
- Processamento em lote de várias páginas.
- Melhor posicionamento de texto para balões inclinados ou muito irregulares.
- Presets de limpeza para mangá preto e branco, webtoon colorido e scans antigos.

## Licença

Defina a licença desejada antes da publicação final.
