# Datathon

Pós graduação em Data Analytics: Projeto final.

Tema: Passos Mágicos

🌐 **Acesse o Dashboard Interativo:** [App Streamlit](https://app-datathon-fiap.streamlit.app/)

Autores:
- **Pedro Lopes de Lucena**
- **Matheus Perez Brassarotto**
- **Douglas Gomes Laranjeira**
- **Marconi Abou Dehn Da Silva Filho**

---

## Estrutura de Arquivos no Repositório

- **`.streamlit/`**: Pasta com arquivos de configuração para a implantação no Streamlit.
- **`aed/`**: Diretório focado em Análise Exploratória de Dados e Modelagem.
  - `01_limpeza_dados_final.ipynb`: Notebook que realiza a limpeza do arquivo original e gera a base limpa.
  - `02_analise_exploratoria_v3.ipynb`: Notebook respondendo as perguntas propostas no projeto, contendo os insights extras no final.
  - `03_modelagem_temporal_risco_defasagem.ipynb`: Notebook com o modelo de predição de risco de defasagem.
- **`app/`**: Diretório da aplicação do dashboard.
  - `main.py`: Arquivo principal contendo o código da aplicação Streamlit.
- **`data/`**: Pasta contendo as bases de dados e arquivos gerados pelo modelo.
  - `BASE DE DADOS PEDE 2024 - DATATHON.xlsx`: Arquivo bruto original da base de dados.
  - `dados_pede.csv`: Base de dados tratada após o processo de limpeza.
  - `modelo_risco_nova_defasagem.joblib`: Modelo de machine learning treinado e exportado.
  - `previsoes_risco_2025.csv`: Base gerada pelo modelo contendo as predições.
- **`requirements.txt`**: Arquivo de texto com as bibliotecas necessárias para executar o projeto e o aplicativo.
- **`README.md`**: Documentação do projeto.

---

## Ferramentas e Tecnologias Utilizadas

- **Python**
- **Streamlit**
- **Scikit-Learn**
- **Pandas**
- **Plotly**
- **Numpy**
- **StatsModel**
- **Joblib**
