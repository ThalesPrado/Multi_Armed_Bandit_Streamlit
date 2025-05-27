import streamlit as st

# 🛠️ Configuração da página (sempre a primeira chamada do Streamlit)
st.set_page_config(page_title="Bandits para Otimização de Preços", layout="wide")

st.title("📘 Teoria, Formulação e Objetivo")

# Objetivo
st.markdown("""
## 🎯 Objetivo do Projeto

O objetivo com a aplicação de algoritmos **Multi-Armed Bandit (MAB)** é **otimizar os preços de venda de produtos** (SKU), maximizando o **lucro ou a receita**, enquanto aprendemos continuamente com o comportamento real dos consumidores.
""")

# Intuição
st.markdown("---")
st.markdown("### 🧠 Intuição: Máquina Caça-Níquel")

st.markdown("""
Imagine que você está em um cassino com várias máquinas caça-níqueis (slot machines). Cada máquina (ou braço) te dá um pagamento (recompensa), mas com uma distribuição aleatória e desconhecida.

Cada **preço** é como uma alavanca de uma máquina caça-níquel. Ao escolher um preço, você observa uma recompensa (lucro) associada à venda a esse preço.

Nosso desafio é o **trade-off** entre:
- **Explorar**: testar outros preços para descobrir novas oportunidades.
- **Explorar o melhor conhecido**: usar o que já sabemos para maximizar o lucro.
""")

# Algoritmo
st.markdown("---")
st.markdown("## ⚙️ Algoritmo")

st.markdown("### 1. Multi-Armed Bandit (Notação Formal)")
st.latex(r"A = \{a_1, a_2, ..., a_K\}")
st.latex(r"r_k \sim \mathcal{D}_k")
st.markdown("""
- Cada ação ak representa um preço.
- rk: recompensa observada ao escolher o preço ak.
- Dk: distribuição de probabilidade (desconhecida) da recompensa.
""")

# Função Objetivo
st.markdown("---")
st.markdown("### 2. Função Objetivo")

st.markdown("Queremos **maximizar a soma esperada das recompensas** ao longo das interações:")

st.latex(r"\text{Maximize} \quad \sum_{t=1}^{T} \mathbb{E}[r_t]")

st.markdown("""
**Onde:**
- T: número total de rodadas (interações).
- rt: recompensa recebida na rodada \( t \).
""")

# Regret
st.markdown("---")
st.markdown("### Equivalentemente: Minimizar o *Regret*")

st.markdown("Como não sabemos de antemão qual ação é a melhor, expressamos o objetivo como:")

st.latex(r"R_T = T \cdot \mu^* - \sum_{t=1}^{T} \mathbb{E}[r_t]")

st.markdown("""
**Onde:**
- mi*: maior recompensa esperada entre todas as ações.
- RT: arrependimento (regret) acumulado até o tempo T.
""")

st.latex(r"""
\boxed{
\text{Maximize } \sum_{t=1}^{T} \mathbb{E}[r_t] \quad \text{ou} \quad \text{Minimize } R_T
}
""")

# Processo de decisão
st.markdown("---")
st.markdown("### 3. Processo de Decisão")

st.markdown(r"""
A cada rodada t:
1. Escolhemos um preço ak.
2. Observamos a recompensa rt (ex: lucro).
3. Atualizamos nossa estimativa para aquele preço.  
""")

# UCB
st.markdown("---")
st.markdown("## 4. 📈 Algoritmo UCB (Upper Confidence Bound)")

st.markdown("""
O algoritmo UCB escolhe o preço com a maior soma entre:
- Média observada até agora.
- Um bônus de incerteza que favorece preços pouco testados.
""")

st.latex(r"a_t = \arg\max_a \left[ \hat{\mu}_a + \sqrt{\frac{2\log t}{n_a}} \right]")

st.markdown("""
- mi_chapeu_a: média da recompensa observada para o preço a  
- na: número de vezes que o preço foi testado  
- t: rodada atual  
""")

# 5 - epsilon-greedy
st.markdown("## 5. 📈 Algoritmo Epsilon-Greedy")

st.markdown("""
O algoritmo **Epsilon-Greedy** busca equilibrar **exploração** e **exploração do melhor preço conhecido** da seguinte forma:

- Com probabilidade ε, escolhe uma ação (preço) aleatoriamente (**exploração**)  
- Com probabilidade 1 - ε, escolhe o preço com maior lucro médio estimado (**exploração do melhor**)  
""")

st.latex(r"""
a_t =
\begin{cases}
\text{ação aleatória}, & \text{com probabilidade } \epsilon \\
\arg\max_a \hat{\mu}_a, & \text{com probabilidade } 1 - \epsilon
\end{cases}
""")

st.markdown("""
- ε: taxa de exploração (ex: 0.1 = 10%)
- mi_chapeu_a: média da recompensa observada para a ação a

Após observar o lucro Ri,t ao testar o preço pi, o algoritmo atualiza sua estimativa de lucro médio Q(pi) da seguinte forma:
""")

st.latex(r"""
Q(p_i) \leftarrow Q(p_i) + \frac{1}{N(p_i)} \left( r_{i,t} - Q(p_i) \right)
""")

st.markdown("""
- Q(pi): estimativa atual do lucro médio para o preço pi
- Ri,t: lucro observado na rodada t com preço pi
- N(pi): número de vezes que o preço pi foi selecionado

Essa fórmula permite que o algoritmo aprenda progressivamente qual preço é mais lucrativo sem precisar armazenar todo o histórico de vendas.
""")


# 6 - Thompson Sampling
st.markdown("## 6. 📈 Algoritmo Thompson Sampling")

st.markdown("""
O algoritmo **Thompson Sampling** é uma abordagem Bayesiana que equilibra automaticamente exploração e exploração.


A cada rodada:
- Ele sorteia um valor de recompensa (lucro) esperado para cada preço com base nas observações anteriores.
- Seleciona o preço que teve a maior amostra sorteada.
- Após a venda, atualiza suas estatísticas, refinando a estimativa futura para aquele preço.

Essa estratégia favorece preços promissores, mas ainda testa preços com alta incerteza, o que é essencial em cenários com dados limitados.

""")

st.latex(r"""
\tilde{\mu}_i \sim \mathcal{N}(\hat{\mu}_i, \hat{\sigma}_i)
""")

st.markdown("""
- mi_chapeu_i: média dos lucros obtidos com o preço pi
- sigma_chapeu_i: desvio padrão (incerteza) do lucro para aquele preço

### Escolha da ação:

""")

st.latex(r"""
a_t = \arg\max_i \tilde{\mu}_i
""")

st.markdown("""
Ou seja, testamos o **preço com a melhor amostra sorteada**.

### Atualização após nova venda:

Após observar o lucro ri,t ao aplicar o preço pi, atualizamos:

- Número de testes N(pi)
- Soma dos lucros S(pi)
- Soma dos quadrados dos lucros SS(pi)

E assim recalculamos:

""")

st.latex(r"""
\hat{\mu}_i = \frac{S(p_i)}{N(p_i)}, \quad
\hat{\sigma}_i = \sqrt{\frac{SS(p_i)}{N(p_i)} - \hat{\mu}_i^2 + \epsilon}
""")


st.markdown("""
O pequeno termo ε evita problemas com variâncias negativas nos primeiros testes.
""")


# Comparação com Regressão
st.markdown("---")
st.markdown("## 7. 📊 Comparação com Modelos de Regressão")

st.markdown("""
| Característica       | Multi-Armed Bandit (MAB) | Regressão Tradicional | Principais diferenças |
|----------------------|---------------------------|------------------------|---------------------|
| **Tipo de problema** | Otimização sequencial    | Previsão / inferência  | Bandits tomam decisões em sequência para maximizar recompensa. Regressão prevê valores com base em variáveis. |
| **Feedback**         | Parcial (só vê a recompensa da ação escolhida) | Completo (vê todos os alvos) | Em MAB, você só vê o lucro do preço testado. Na regressão, você conhece todos os alvos. |
| **Exploração**       | Sim (via epsilon-Greedy, UCB) | Não (base fixa) | Bandits testam novas ações mesmo sem certeza. Regressão só aprende com dados que já possui. |
| **Atualização**      | Online (a cada nova interação) | Offline (batch completo) | MAB aprende em tempo real. Regressão precisa reprocessar toda a base para atualizar. |
| **Função objetivo**  | Maximizar soma de recompensas ou minimizar o arrependimento (regret) | Minimizar erro entre previsão e valor real (erro quadrático médio) | Bandits otimizam decisões, regressão minimiza diferença entre previsto e real. |
| **Suposições**       | Não assume forma funcional dos dados | Assume forma funcional (linearidade, normalidade, etc.) | Bandits são mais flexíveis. Regressão exige estrutura estatística conhecida. |
| **Aplicações**       | Otimização de preços, testes A/B, marketing online | Previsão de vendas, risco de crédito, churn | MAB é melhor para aprendizado ativo. Regressão serve para estimativa e interpretação. |
""", unsafe_allow_html=True)

# Limitações
st.markdown("---")
st.markdown("## ⚠️ Limitações do Modelo")

st.markdown("""
- Pode ter desempenho fraco no início (exploração aleatória),nos primeiros passos ele ainda não tem informações suficientes para saber qual preço é melhor. Ele testa aleatoriamente o que pode gerar escolhas ruins no começo.

- Modelos simples como UCB não consideram variáveis contextuais, ignora fatores como dia da semana, canal de venda, promoções ativas e etc. O preço ideal em um contexto pode ser diferente em outro contexto.

- Requer número razoável de interações para convergir. Se houver poucos dados pode escolher preços ruins por sorte, se você só testou o preço R$ 9 duas vezes e ele deu lucro alto, o algoritmo ainda não sabe se foi sorte ou se é o melhor preço mesmo.

- Assume que as recompensas não mudam com o tempo (estacionariedade),assume que o comportamento do cliente e o lucro por preço permanecem constantes ao longo do tempo.
""")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import os

# -------------------
# ANÁLISE DE DADOS
# -------------------

st.markdown("---")
st.title("📦 Visão Geral dos Produtos (SKU) + Dispersão de Lucro")

uploaded_file = st.file_uploader("📂 Faça upload da base histórica (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Conversão segura para colunas numéricas
    cols_to_convert = ['price', 'unit_cost', 'demand', 'total_profit', 'total_revenue', 'unit_profit']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    st.subheader("📋 Prévia da Base de Dados")
    st.dataframe(df.head())

    # ---------------------
    # INDICADORES POR PRODUTO
    # ---------------------
    resumo_por_sku = df.groupby(['sku_id', 'sku_name']).agg(
        preco_medio=('price', 'mean'),
        custo_medio=('unit_cost', 'mean'),
        lucro_medio_por_unidade=('unit_profit', 'mean'),
        demanda_total=('demand', 'sum'),
        faturamento_total=('total_revenue', 'sum'),
        lucro_total= ('total_profit', 'sum'),
        dias_registrados=('sku_id', 'count')
    ).reset_index()

    st.subheader("📊 Indicadores Agregados por SKU")
    st.dataframe(resumo_por_sku)

    # ---------------------
    # BOX PLOT DOS LUCROS
    # ---------------------
    st.subheader("📈 Dispersão dos Lucros Diários por Produto")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='sku_name', y='total_profit', palette='viridis', ax=ax)
    sns.stripplot(data=df, x='sku_name', y='total_profit', color='#b8860b', alpha=0.8, jitter=0.2, ax=ax)
    ax.set_title("Boxplot + Pontos de Lucro Diário por Produto (SKU)")
    ax.set_ylabel("Lucro por Observação (R$)")
    ax.set_xlabel("Produto (SKU)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # ---------------------
    # OTIMIZAÇÃO DE PREÇO COM MAB
    # ---------------------
    st.markdown("---")
    st.markdown("## 🤖 Otimização com Multi-Armed Bandits")

# ---------------------
# SELEÇÃO DE SKU
# ---------------------
    st.subheader("🔍 Análise por SKU e Algoritmos MAB")
    sku_opcoes = df['sku_name'].unique()
    sku_selecionado = st.selectbox("Escolha o SKU:", options=sku_opcoes, key="sku_otimizacao")
    df_sku = df[df['sku_name'] == sku_selecionado].copy()

    df_sku['reward'] = df_sku['total_profit']
    df_sku['arm'] = df_sku['price'].round(2)

# Inicialmente define os preços e braços com base nos dados reais
    price_options = sorted(df_sku['arm'].unique())
    n_arms = len(price_options)
    arm_map = {price: idx for idx, price in enumerate(price_options)}
    df_sku['arm_index'] = df_sku['arm'].map(arm_map)

    class UCBBandit:
        def __init__(self, n):
            self.n = n
            self.counts = np.zeros(n)
            self.values = np.zeros(n)
            self.total = 0

        def select_arm(self):
            self.total += 1
            if 0 in self.counts:
                return np.argmin(self.counts)
            return np.argmax(self.values + np.sqrt(2 * np.log(self.total) / self.counts))

        def update(self, arm, reward):
            self.counts[arm] += 1
            n = self.counts[arm]
            value = self.values[arm]
            self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    class EpsilonGreedyBandit:
        def __init__(self, n, epsilon=0.1):
            self.n = n
            self.epsilon = epsilon
            self.counts = np.zeros(n)
            self.values = np.zeros(n)

        def select_arm(self):
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n)
            return np.argmax(self.values)

        def update(self, arm, reward):
            self.counts[arm] += 1
            n = self.counts[arm]
            value = self.values[arm]
            self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    class ThompsonSamplingBandit:
        def __init__(self, n):
            self.n = n
            self.counts = np.zeros(n)
            self.sum_rewards = np.zeros(n)
            self.sum_squares = np.zeros(n)

        def select_arm(self):
            samples = []
            for i in range(self.n):
                if self.counts[i] < 2:
                    samples.append(np.random.normal(0, 100))
                else:
                    mean = self.sum_rewards[i] / self.counts[i]
                    std = np.sqrt(self.sum_squares[i] / self.counts[i] - mean ** 2 + 1e-6)
                    samples.append(np.random.normal(mean, std))
            return np.argmax(samples)

        def update(self, arm, reward):
            self.counts[arm] += 1
            self.sum_rewards[arm] += reward
            self.sum_squares[arm] += reward ** 2

    ucb = UCBBandit(n_arms)
    eps = EpsilonGreedyBandit(n_arms)
    ts = ThompsonSamplingBandit(n_arms)
    ucb_total = eps_total = ts_total = 0
    ucb_rewards = []
    eps_rewards = []
    ts_rewards = []

    for _, row in df_sku.iterrows():
        ucb_arm = ucb.select_arm()
        reward_ucb = row['reward'] if row['arm_index'] == ucb_arm else 0
        ucb.update(ucb_arm, reward_ucb)
        ucb_total += reward_ucb
        ucb_rewards.append(ucb_total)

        eps_arm = eps.select_arm()
        reward_eps = row['reward'] if row['arm_index'] == eps_arm else 0
        eps.update(eps_arm, reward_eps)
        eps_total += reward_eps
        eps_rewards.append(eps_total)

        ts_arm = ts.select_arm()
        reward_ts = row['reward'] if row['arm_index'] == ts_arm else 0
        ts.update(ts_arm, reward_ts)
        ts_total += reward_ts
        ts_rewards.append(ts_total)

    st.subheader("📈 Lucro Acumulado das Estratégias")
    st.line_chart({"UCB": ucb_rewards, "Epsilon-Greedy": eps_rewards, "Thompson Sampling": ts_rewards})

    st.markdown(f"💰 **Lucro total com UCB**: R$ {ucb_total:,.2f}")
    st.markdown(f"💰 **Lucro total com Epsilon-Greedy**: R$ {eps_total:,.2f}")
    st.markdown(f"💰 **Lucro total com Thompson Sampling**: R$ {ts_total:,.2f}")

    melhor_preco_ucb = price_options[np.argmax(ucb.values)]
    melhor_preco_eps = price_options[np.argmax(eps.values)]
    melhor_preco_ts = price_options[np.argmax(ts.sum_rewards / (ts.counts + 1e-6))]

    st.subheader("🏆 Preços Otimizados Aprendidos")
    st.markdown(f"- 🎯 Melhor preço segundo **UCB**: R$ {melhor_preco_ucb}")
    st.markdown(f"- 🎯 Melhor preço segundo **Epsilon-Greedy**: R$ {melhor_preco_eps}")
    st.markdown(f"- 🎯 Melhor preço segundo **TS**: R$ {melhor_preco_ts}")

    def simular_lucro_estimado(df_base, preco):
        media_lucro = df_base[df_base['arm'] == preco]['reward'].mean()
        return media_lucro * len(df_base)

    lucro_ucb_estimado = simular_lucro_estimado(df_sku, melhor_preco_ucb)
    lucro_eps_estimado = simular_lucro_estimado(df_sku, melhor_preco_eps)
    lucro_ts_estimado = simular_lucro_estimado(df_sku, melhor_preco_ts)

    st.subheader("📈 Comparativo Estimado de Estratégias")
    st.markdown(f"- 💼 Estimado com preço **UCB** ({melhor_preco_ucb:.2f}): **R$ {lucro_ucb_estimado:,.2f}**")
    st.markdown(f"- 💼 Estimado com preço **Epsilon-Greedy** ({melhor_preco_eps:.2f}): **R$ {lucro_eps_estimado:,.2f}**")
    st.markdown(f"- 💼 Estimado com preço **Thompson Sampling** ({melhor_preco_ts:.2f}): **R$ {lucro_ts_estimado:,.2f}**")
    st.markdown(f"- 💼 **Lucro total histórico** (todas as vendas): **R$ {df_sku['total_profit'].sum():,.2f}**")
    # ---------------------
    # GRÁFICO DE LUCRO MÉDIO POR PREÇO PRATICADO
    # ---------------------
    st.subheader("📊 Lucro Médio por Preço Praticado no Histórico")
    lucro_por_preco = df_sku.groupby('arm')['reward'].mean().sort_index()
    st.bar_chart(lucro_por_preco)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    estrategias = ['Histórico', 'Estimado UCB', 'Estimado Epsilon','Estimado TS']
    valores = [df_sku['total_profit'].sum(), lucro_ucb_estimado, lucro_eps_estimado,lucro_ts_estimado]
    ax3.bar(estrategias, valores, color=['gray', 'green', 'blue','yellow'])
    ax3.set_ylabel("Lucro Total Estimado (R$)")
    ax3.set_title(" Comparativo Estimado de Estratégias")
    for i, v in enumerate(valores):
        ax3.text(i, v + 0.01 * max(valores), f'R$ {v:,.0f}', ha='center')
    st.pyplot(fig3)

    # ---------------------
    # ANALISES ADICIONAIS DE REGRET
    # ---------------------
    st.subheader("📉 Regret e Eficiência de Preço")
    lucro_medio_por_arm = df_sku.groupby('arm')['reward'].mean()
    preco_otimo = lucro_medio_por_arm.idxmax()
    lucro_ideal_unitario = lucro_medio_por_arm.max()
    lucro_real_total = df_sku['reward'].sum()
    T = len(df_sku)
    lucro_ideal_total = T * lucro_ideal_unitario
    regret_total = lucro_ideal_total - lucro_real_total

    st.markdown(f"- 🧠 **Preço ótimo estimado (lucro médio)**: R$ {preco_otimo:.2f}")
    st.markdown(f"- 📉 **Regret acumulado (total perdido por não aplicar o preço ótimo)**: R$ {regret_total:,.2f}")

    st.markdown(f"Neste caso, o algoritmo estima que a empresa poderia ter ganho {regret_total:,.2f} a mais se tivesse vendido todas as unidades ao preço ótimo de {preco_otimo:.2f}. Esse valor {preco_otimo:.2f} é o preço ótimo estimado com base no histórico de vendas, definido como o preço que gerou, em média, o maior lucro por observação (venda) no dataset.")

    # Gráfico de regret acumulado
    regret_series = np.cumsum([lucro_ideal_unitario - r for r in df_sku['reward']])
    st.line_chart(regret_series)

    # Regret médio por preço praticado
    df_sku['regret_unitario'] = lucro_ideal_unitario - df_sku['reward']
    st.subheader("📊 Regret Médio por Preço Praticado")
    chart_data = df_sku.groupby('price')['regret_unitario'].mean().sort_index()
    st.bar_chart(chart_data)

    st.markdown("Regret médio = (Lucro médio com preço ótimo) – (Lucro médio com preço praticado). Aqui podemos medir o quanto de lucro deixamos de ganhar, em média, ao vender a esse preço específico em vez do preço ótimo.")

    # Ranking de SKUs por regret total
    regret_por_sku = df.groupby('sku_name').apply(
        lambda x: (lucro_ideal_unitario - x['total_profit'].mean()) * len(x)
    ).sort_values(ascending=False).reset_index()
    regret_por_sku.columns = ['sku_name', 'regret_estimado']

    st.subheader("🏷️ Ranking de Produtos por Regret Estimado")
    st.dataframe(regret_por_sku)

    st.markdown("Este ranking mostra quanto cada produto deixou de lucrar, em comparação com o preço ótimo estimado.Produtos com maior regret estimado indicam oportunidades de otimização de preço.")

# 📊 Visualização de Incerteza (Thompson Sampling)
# ---------------------
    st.subheader("📉 Incerteza Estimada por Preço (Thompson Sampling)")

    fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
    x = np.linspace(df_sku['reward'].min() * 0.9, df_sku['reward'].max() * 1.1, 200)

    # Apenas preços com no mínimo 5 observações
    precos_confiaveis = [i for i in range(n_arms) if ts.counts[i] >= 5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(precos_confiaveis)))

    for j, i in enumerate(precos_confiaveis):
        mean = ts.sum_rewards[i] / ts.counts[i]
        var = ts.sum_squares[i] / ts.counts[i] - mean ** 2
        std = np.sqrt(max(var, 1e-6))
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax_ts.plot(x, pdf, label=f"{price_options[i]:.2f}", color=colors[j], alpha=0.8)

    ax_ts.set_title("Distribuição Posterior das Recompensas Estimadas (TS)")
    ax_ts.set_xlabel("Lucro Estimado")
    ax_ts.set_ylabel("Densidade")
    ax_ts.legend(title="Preço", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_ts)

    st.markdown("""

- Cada curva no gráfico representa a distribuição da incerteza do lucro estimado para um preço específico.

- Essas curvas foram geradas pelo algoritmo Thompson Sampling, que assume que o lucro de cada preço segue uma distribuição normal com média e variância estimadas a partir dos dados.

- Os picos das curvas indicam o lucro médio estimado por unidade vendida para cada preço testado.

- **Curva Estreita**: Algoritmo tem alta confiança no lucro estimado (muitos testes, dados consistentes).  

- **Curva Ampla ou Achatadas**: Alta incerteza sobre o lucro (poucos testes ou alta variabilidade nos resultados).

- Priorizar preços com:
    - Alta média estimada 
    - Baixa variância (curvas bem definidas)

""")

    st.markdown("---")
    st.subheader("🔍 Comparação Visual: Curvas Estreitas vs. Curvas Amplas (TS)")

    # Exemplo fictício de distribuições normais
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1000, 500)

    # Curva concentrada (baixa incerteza)
    mu_baixa_inc = 600
    sigma_baixa_inc = 30
    y1 = (1 / (sigma_baixa_inc * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_baixa_inc) / sigma_baixa_inc) ** 2)

    # Curva ampla (alta incerteza)
    mu_alta_inc = 600
    sigma_alta_inc = 150
    y2 = (1 / (sigma_alta_inc * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_alta_inc) / sigma_alta_inc) ** 2)

    # Plot
    fig_exemplo, ax_exemplo = plt.subplots(figsize=(10, 4))
    ax_exemplo.plot(x, y1, label="Curva Estreita (Alta Confiança)", color='green')
    ax_exemplo.plot(x, y2, label="Curva Ampla (Alta Incerteza)", color='red')
    ax_exemplo.set_title("Exemplo Comparativo: Distribuições Posteriores (TS)")
    ax_exemplo.set_xlabel("Lucro Estimado")
    ax_exemplo.set_ylabel("Densidade")
    ax_exemplo.legend()
    st.pyplot(fig_exemplo)

# ---------------------
# 📋 Tabela com Intervalo de Confiança (95%)
# ---------------------
    st.subheader("📋 Estimativas com Intervalo de Confiança (95%) - Thompson Sampling")

    dados_ts = []
    for i, preco in enumerate(price_options):
        if ts.counts[i] >= 5:
            mu = ts.sum_rewards[i] / ts.counts[i]
            var = ts.sum_squares[i] / ts.counts[i] - mu ** 2
            std = np.sqrt(max(var, 1e-6))
            ci_low, ci_up = stats.norm.interval(0.95, loc=mu, scale=std)
            dados_ts.append([preco, mu, std, ci_low, ci_up, int(ts.counts[i])])

    df_ts = pd.DataFrame(dados_ts, columns=[
        "Preço", "Média Estimada", "Desvio Padrão", "CI Inferior", "CI Superior", "N Testes"
    ])

    st.dataframe(df_ts.style.format({
        "Preço": "{:.2f}",
        "Média Estimada": "{:.2f}",
        "Desvio Padrão": "{:.2f}",
        "CI Inferior": "{:.2f}",
        "CI Superior": "{:.2f}"
    }))

    st.markdown("""
A tabela acima mostra, para cada preço testado, a média estimada de lucro, o nível de incerteza (via desvio padrão), o intervalo de confiança de 95%, e o número de testes realizados.

- Preço: valor do produto testado.

- Média Estimada: lucro médio observado ao vender com esse preço.

- Desvio Padrão: grau de incerteza nas estimativas. Quanto menor, mais confiável.

- CI Inferior / CI Superior: intervalo de confiança onde, com 95% de certeza, o verdadeiro lucro médio está contido.

- N Testes: número de vezes que o preço foi testado (impacta a confiabilidade).

- Mesmo que a média estimada pareça alta, um desvio padrão elevado pode indicar que é um resultado influenciado por variabilidade/sorte.


    """)
