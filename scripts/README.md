# Scripts Directory

Este diretório contém todos os scripts executáveis do projeto.

## 📁 Organização

### Scripts Principais (Pipeline Completo)

1. **`generate_dataset.py`** - Gera dataset com expert DQN
   - Executa agente DQN no ambiente
   - Salva imagens e ações em CSV
   - Output: `dataset_big_highway/`

2. **`encode_images.py`** - Codifica imagens com CLIP
   - Processa imagens do dataset
   - Gera embeddings .npy
   - Usa processamento em batch para eficiência

3. **`train_vlm.py`** - Treina modelo VLM (VERSÃO FINAL)
   - Dataset: 200k frames
   - Modelo: Qwen-0.6B + LoRA
   - Output: `vlm_v3.pth`

4. **`evaluate.py`** - Avalia modelo treinado
   - Métricas: Acurácia, Top-3, Matriz de confusão
   - Processa imagens reais com CLIP
   - Análise por classe

5. **`demo.py`** - Demonstração visual
   - Mostra VLM dirigindo em tempo real
   - Usa OpenCV para visualização
   - Controle: 'q' para sair

### Scripts Legados (Referência)

- **`train_vlm_v1.py`** - Primeira versão do treinamento (esqueleto)
- **`train_vlm_v2.py`** - Versão com dataset menor (16k)
- **`train_vlm_200k.py`** - Versão alternativa para 200k
- **`a_old.py`** - Código antigo/experimental

### Scripts de Treinamento RL

Localizados em `src/rl/`:
- **`src/rl/train.py`** - Treina agente DQN expert

## 🚀 Ordem de Execução

Para reproduzir o projeto completo:

```bash
# Passo 1: Treinar DQN (opcional, já temos modelo)
cd src/rl && python train.py

# Passo 2-6: Pipeline VLM
cd ../..
python scripts/generate_dataset.py
python scripts/encode_images.py
python scripts/train_vlm.py
python scripts/evaluate.py
python scripts/demo.py
```

## ⚙️ Configurações

Cada script tem configurações no topo do arquivo:

```python
# Exemplo: train_vlm.py
CSV_PATH = "dataset_big_highway/dataset_highway_200k.csv"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
```

Edite estas variáveis conforme necessário.

## 📝 Notas

- Scripts marcados com `_v1`, `_v2` são versões antigas
- Use sempre as versões sem sufixo para reprodução
- Certifique-se de estar na raiz do projeto ao executar
