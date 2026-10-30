# Dashboard de Vendas de Jogos

Trabalho de Estatística e Probabilidade (4º semestre). Dashboard em Streamlit que analisa vendas de jogos eletrônicos (1980–2016, dataset `vgsales.csv`).

## Funcionalidades
- Análises descritivas: Top 10 jogos, plataformas, gêneros e vendas por região.
- Previsão de vendas do próximo ano com o **Método de Holt** (séries temporais, tendência).
- Validação por **Backtesting Temporal**: oculta o último ano, treina com o passado e mede o erro (MAE).

## Como executar
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tecnologias
Python, Streamlit, Pandas, Plotly, Statsmodels.
