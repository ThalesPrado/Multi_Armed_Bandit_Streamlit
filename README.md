# 🎯 Otimização de Preços com Multi-Armed Bandits (MAB)

Este projeto utiliza algoritmos de **Aprendizado por Reforço** para **otimizar preços de venda de produtos (SKUs)** com base em dados históricos de lucro. A aplicação permite simular estratégias como **UCB**, **Epsilon-Greedy** e **Thompson Sampling**, ajudando analistas e equipes de pricing a tomar decisões mais eficientes em ambientes com incerteza.

## 📌 Objetivo

> Maximizar o lucro de vendas de diferentes produtos por meio de aprendizado sequencial, aprendendo com o comportamento real dos consumidores ao longo do tempo.

## 🧠 Algoritmos implementados

- **UCB (Upper Confidence Bound)**  
- **Epsilon-Greedy**  
- **Thompson Sampling** (com visualização bayesiana das distribuições posteriores)

## 🛠️ Funcionalidades

- Upload da base histórica de vendas
- Visualização por SKU: lucro, dispersão, estratégias
- Lucro acumulado por estratégia
- Preço ótimo aprendido
- Comparativo estimado entre estratégias
- Análise de *regret* e incerteza
- Intervalos de confiança para os preços testados

## 📊 Exemplo de uso

A aplicação permite:
- Analisar produtos individualmente
- Simular dados realistas para cenários com poucas observações
- Comparar visualmente as decisões tomadas por cada algoritmo
- Estimar impacto de adoção de políticas de preço otimizadas

## ▶️ Como executar

### Requisitos
streamlit
pandas
numpy
matplotlib
seaborn
scipy
openpyxl

### Instale as dependências:

```bash
pip install -r requirements.txt
