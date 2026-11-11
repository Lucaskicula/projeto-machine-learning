# Reconhecimento de Pokémon em vídeo

Projeto preparado a partir dos requisitos da atividade. As classes escolhidas são `bulbasaur`, `charmander`, `squirtle` e `fundo`; os nomes das pastas definem as classes.

## 1. Gravar os dados

Crie pelo menos quatro vídeos diferentes para cada Pokémon e quatro para `fundo`. Grave cada classe em cenários claros e escuros, misturando frente, costas, os dois lados, cima, baixo, perto e longe em cada vídeo. O conjunto de teste (30 a 60 segundos) deve ser uma gravação nova, em outro local. Evite que cada Pokémon apareça sempre com um cenário próprio.

Organize os arquivos assim:

```text
videos/
  bulbasaur/  v01_claro.mp4 ...
  charmander/
  squirtle/
  fundo/
```

## 2. Instalar e executar

Com Python 3.12, crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
```

No Windows, instale e execute os scripts assim:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe extrair_frames.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe detectar.py teste_novo.mp4
```

No Linux ou macOS, use o Python do ambiente virtual (`.venv/bin/python`) nos mesmos comandos.

O repositório inclui as gravações em `videos/`, os frames já extraídos em `data/` e o modelo treinado. Se adicionar novas gravações, rode novamente `extrair_frames.py` e `train.py`.

O treino salva `modelo.keras`, `class_names.json`, `curvas.png` e `dataset_imagens.pdf`; a detecção salva `saida_teste_novo.mp4` e imprime a linha do tempo.

## 3. Melhorias da atividade

- `detectar.py` calcula a média de probabilidade dos últimos oito frames e exibe `?` se a confiança média for menor que 60%. Os valores ficam no início do arquivo para experimentação.
- A linha do tempo agrupa rótulos consecutivos, removendo fundo, desconhecidos e trechos menores que meio segundo.
- `train.py` treina primeiro a cabeça com a MobileNetV2 congelada, depois libera até 30 camadas finais com taxa `1e-5`. O gráfico marca o início dessa fase.
- A validação separa vídeos inteiros, nunca frames aleatórios do mesmo vídeo.

## 4. Entrega e experimentos

O vídeo anotado `saida_testee.mp4`, o PDF do dataset e as curvas geradas estão incluídos como resultados desta execução. Ainda faltam o print da linha do tempo e o relatório com as cinco respostas da Parte 5. Para responder às comparações, rode as variações pedidas (split por foto, learning rate `1e-3`, ângulo omitido, quarto Pokémon e dois objetos no quadro) e registre os resultados observados.
