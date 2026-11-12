# Reconhecimento de Pokémon em vídeo

Projeto preparado a partir dos requisitos da atividade. As classes são `bulbasaur`, `charmander`, `squirtle` e `fundo`; os nomes das pastas definem as classes.

## Gravar e organizar os dados

O conjunto contém vídeos por Pokémon, incluindo frente, costas, cima, baixo e aproximações, além de gravações da classe `fundo`. Os vídeos e os frames extraídos ficam em `videos/` e `data/`.

```text
videos/
  bulbasaur/
  charmander/
  squirtle/
  fundo/
```

Ao acrescentar gravações, rode a extração e o treino novamente.

## Instalar e executar

Com Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe extrair_frames.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe detectar.py teste_novo.mp4
```

O treino salva `modelo.keras`, `class_names.json`, `curvas.png` e `dataset_imagens.pdf`. A detecção salva um vídeo anotado e imprime a linha do tempo.

## Como o modelo funciona

- `detectar.py` calcula a média das probabilidades dos últimos oito frames e mostra `?` quando a confiança média fica abaixo de 60%.
- A linha do tempo agrupa rótulos consecutivos, removendo fundo, desconhecidos e trechos menores que meio segundo.
- `train.py` treina a cabeça com a MobileNetV2 congelada e depois libera até 30 camadas finais com taxa de aprendizado `1e-5`.
- A validação separa vídeos inteiros, evitando misturar frames do mesmo vídeo entre treino e validação.

## Experimentos e entrega

O relatório em `output/pdf/relatorio_respostas_pokemon.pdf` responde às cinco perguntas da atividade com os experimentos e os limites observados. O teste de costas excluiu os frames de `bulbasaur_costas_video` do treino experimental; o modelo reconheceu Bulbasaur nesse clipe. O vídeo do Venusaur, que não estava entre as classes treinadas, foi classificado como Bulbasaur com confiança acima do limiar, ilustrando a limitação do softmax e da rejeição por confiança.

- `experimentos/q3_bulbasaur_sem_costas/`: modelo experimental, classes e resultados numéricos.
- `experimentos/q4_venusaur/venusaur.mp4`: vídeo de Pokémon desconhecido usado no teste.
- `saida_bulbasaur_costas_video.mp4` e `saida_venusaur.mp4`: vídeos anotados dos testes.
- `saida_teste-pokemons.mp4`: demonstração com vários Pokémon visíveis; o classificador retorna um rótulo por quadro, não caixas individuais.

As métricas são válidas para os vídeos usados e não garantem o mesmo resultado em qualquer gravação.
