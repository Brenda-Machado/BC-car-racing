# Pesquisa PIBIC - Aprendizado por Imitação com Multimodalidade de Ações para Navegação de Veı́culos Autônomos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.21.0-green.svg)](https://gym.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

Este repositório contém as implementações referentes à pesquisa desenvolvida em torno do tema "Aprendizado por Imitação com Multimodalidade de Ações para
Navegação de Veı́culos Autônomos", realizada entre 2024/2025 no programa PIBIC. Foi desenvolvidos implementações de Behavior Cloning (BC) e Implicit Behavior Cloning (IBC) para condução autônoma utilizando Redes Neurais Convolucionais (CNNs), no ambiente CarRacing-v0 do OpenAI Gym.

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura Técnica](#arquitetura-técnica)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Coleta de Dados](#coleta-de-dados)
- [Avaliação e Visualização](#avaliação-e-visualização)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração Avançada](#configuração-avançada)
- [Referências](#referências)
- [Contribuições](#contribuições)

## Visão Geral

Behavior Cloning é uma abordagem de aprendizado supervisionado para aprendizado por imitação em que uma rede neural aprende a replicar comportamentos de especialistas através de dados de demonstração. Esta implementação foca em condução autônoma, onde:

- **Entrada**: Imagens RGB do ambiente de simulação
- **Saída**: Ações de controle contínuas `[direção, freio, acelerador]`
- **Paradigma de Aprendizado**: Aprendizado supervisionado com trajetórias de especialistas

### Características Principais

- Arquitetura baseada em CNN para percepção visual
- Empilhamento temporal de frames para compreensão dinâmica
- Interface de coleta de dados de especialistas
- Pipeline abrangente de treinamento e avaliação
- Ferramentas de visualização de performance

## Arquitetura Técnica

### Design da Rede Neural

O modelo implementa uma Rede Neural Convolucional, baseada na arquitetura proposta por [Irving et al. (2023)](https://repositorio.ufsc.br/handle/123456789/251825).

```
Entrada: 84x84x4 (tons de cinza, pilha de 4 frames)
    ↓
Camadas Conv2D + Batch Normalization + ReLU
    ↓
Global Average Pooling
    ↓
Camadas Totalmente Conectadas
    ↓
Saída: 3 ações contínuas [direção, freio, acelerador]
```

### Pipeline de Pré-processamento de Dados

1. **Pré-processamento de Imagem**:
   - Converter frames RGB para tons de cinza (96x96) -> (84x84)
   - Normalizar valores de pixels para [0,1]
   - Aplicar empilhamento temporal (4 frames consecutivos)

2. **Espaço de Ações**:
   - Direção: [-1, 1] (esquerda/direita)
   - Freio: [0, 1] (sem freio/freio total)
   - Acelerador: [0, 1] (sem aceleração/aceleração total)

## Configuração do Ambiente

### Pré-requisitos

- Python 3.8+

### Início Rápido

Navegue até o diretório `bc-trainer` e execute:

```bash
make run
```

Este comando irá:
1. Criar um ambiente virtual Python
2. Instalar todas as dependências necessárias
3. Executar o treinamento (`main.py`)
4. Gerar métricas de avaliação e visualizações

### Instalação Manual

```bash
# Criar ambiente virtual
python -m venv bc_env
source bc_env/bin/activate  
# bc_env\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Processo de Treinamento

1. Carregar trajetórias de especialistas de `bc-trainer/data/trajectories/`
2. Dividir dados em conjuntos de treino/validação (80/20)
3. Treinar CNN com otimizador Adam
4. Salvar melhor modelo baseado na perda de validação
5. Gerar métricas de performance

## Avaliação e Visualização

### Métricas de Performance

```bash
make plot
```

### Código expandido do IBC

Foi implementado aprendizado por imitação no repositório "ibc-trainer-bruno", o qual diz respeito a um código desenvolvido por @Bruno e expandido por @Brenda-Machado durante a pesquisa PIBIC. A expansão diz respeito à implementação de Modelos baseados em Energia (EBMs) para aprendizado do CarRacing.

### Trabalhos relacionados

- [Tutorial de Behavior Cloning no CarRacing](https://github.com/Brenda-Machado/simple-bc-tutorial)

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Autores

- **Brenda Machado** - *Bolsista PIBIC* - [Brenda-Machado](https://github.com/Brenda-Machado)
- **Eric Aislan Antonelo** - *Orientador*
- **Bruno**

## Agradecimentos

Este projeto foi desenvolvido com apoio do Programa Institucional de Bolsas de Iniciação Científica (PIBIC), que proporcionou os recursos e orientação necessários para a realização desta pesquisa.

