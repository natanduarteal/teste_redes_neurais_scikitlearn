import streamlit as pd_st # Usando o alias alternativo apenas para importação organizada se necessário, mas importaremos o padrão abaixo
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(
    page_title="Portal de IA - Diagnóstico de Serviços",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para visual profissional
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e293b;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://placehold.co/400x150/1e293b/ffffff?text=Portal+IA+Serviços", use_container_width=True)
    st.title("Instruções de Uso")
    st.write("""
    Este portal utiliza uma **Rede Neural Multi-Layer Perceptron (MLPClassifier)** para aprender os padrões operacionais da sua empresa e diagnosticar quais serviços necessitam de treinamento prioritário.
    """)
    
    st.subheader("Layout do CSV Esperado:")
    st.code("""id_servico,nome_servico,reclamacoes_mensais,tempo_resolucao_horas,nota_satisfacao""")
    
    # Gerando dados de demonstração para download rápido
    dados_exemplo = pd.DataFrame({
        "id_servico": [1, 2, 3, 4, 5],
        "nome_servico": ["Suporte Técnico", "Setor Financeiro", "Logística de Entrega", "Atendimento Comercial", "Pós-Venda"],
        "reclamacoes_mensais": [12, 45, 80, 8, 55],
        "tempo_resolucao_horas": [8.5, 48.0, 96.5, 6.0, 75.0],
        "nota_satisfacao": [4.8, 3.2, 1.8, 4.6, 2.1]
    })
    
    csv_exemplo = dados_exemplo.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar CSV de Exemplo",
        data=csv_exemplo,
        file_name="exemplo_metricas_servicos.csv",
        mime="text/csv"
    )

st.title("🤖 Diagnóstico de Desempenho e Necessidade de Treinamento por IA")
st.markdown("---")

col_upload, col_preview = st.columns([1, 2])

with col_upload:
    st.subheader("1. Importação de Dados")
    arquivo_carregado = st.file_uploader(
        "Carregue as métricas operacionais dos serviços em formato CSV",
        type=["csv"]
    )

df = None

if arquivo_carregado is not None:
    try:
        # Tenta ler com detecção automática do delimitador
        df = pd.read_csv(arquivo_carregado, sep=None, engine='python', encoding='utf-8')
        
        # Corrige caractere BOM no início se houver
        if df.columns[0] == '\ufeffid_servico':
            df.rename(columns={'\ufeffid_servico': 'id_servico'}, inplace=True)
            
        colunas_obrigatorias = [
            "id_servico", 
            "nome_servico", 
            "reclamacoes_mensais", 
            "tempo_resolucao_horas", 
            "nota_satisfacao"
        ]
        
        # Fallback de codificação se falhar
        if not all(col in df.columns for col in colunas_obrigatorias):
            arquivo_carregado.seek(0)
            df = pd.read_csv(arquivo_carregado, sep=None, engine='python', encoding='latin1')
            if df.columns[0] == '\ufeffid_servico':
                df.rename(columns={'\ufeffid_servico': 'id_servico'}, inplace=True)
                
        # Validação rígida
        if not all(col in df.columns for col in colunas_obrigatorias):
            st.error("O arquivo CSV precisa conter as colunas exatas: id_servico, nome_servico, reclamacoes_mensais, tempo_resolucao_horas, nota_satisfacao")
            df = None
            
        if df is not None:
            df.dropna(subset=colunas_obrigatorias, inplace=True)
            st.success(f"Arquivo carregado com sucesso! Mapeados {len(df)} registros.")
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {str(e)}")

# Exibe pré-visualização se o df existir
with col_preview:
    st.subheader("Pré-visualização dos Dados Carregados")
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aguardando upload do arquivo CSV para pré-visualização.")

if df is not None:
    st.markdown("---")
    st.subheader("2. Executar Inteligência Artificial")
    
    if st.button("Ativar Análise com Rede Neural (MLPClassifier)"):
        with st.spinner("Treinando rede neural e calculando diagnósticos..."):
            try:
                # Função de rotulação semântica heurística (para aprendizado supervisionado)
                def rotular_desempenho(row):
                    nota = float(row["nota_satisfacao"])
                    tempo = float(row["tempo_resolucao_horas"])
                    reclamacoes = float(row["reclamacoes_mensais"])
                    
                    if nota >= 4.5 and tempo <= 12 and reclamacoes < 15:
                        return "serviço excelente"
                    elif nota >= 3.5 and tempo <= 48:
                        return "serviço bom"
                    elif nota >= 2.0 or tempo > 72 or reclamacoes > 50:
                        return "serviço ruim"
                    else:
                        return "serviço muito ruim"

                # Criando vetor de treino
                df["classe_alvo"] = df.apply(rotular_desempenho, axis=1)

                features = ["reclamacoes_mensais", "tempo_resolucao_horas", "nota_satisfacao"]
                X_dados = df[features].values
                y_dados = df["classe_alvo"].values

                # Normalização Estatística
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_dados)

                # Inicialização e treinamento da Rede Neural
                rede_neural = MLPClassifier(
                    hidden_layer_sizes=(16, 8), 
                    activation='relu',
                    max_iter=500, 
                    random_state=42
                )
                rede_neural.fit(X_scaled, y_dados)

                # Predição Final
                df["resultado_ia"] = rede_neural.predict(X_scaled)

                # Apresentando Resultados formatados
                st.subheader("🎯 Diagnóstico Final de Desempenho e Treinamento")
                
                # Exibindo como cards coloridos usando colunas responsivas do Streamlit
                col_res1, col_res2 = st.columns([1, 1])
                
                relatorios_lista = []
                for nome_servico, registros in df.groupby("nome_servico"):
                    pior_classe = registros["resultado_ia"].mode()[0] if not registros["resultado_ia"].empty else "Não classificado"
                    
                    # Definição de diagnóstico e cores para destaque visual
                    if pior_classe == "serviço muito ruim":
                        cor = "🔴"
                        diagnostico = "SERVIÇO MUITO RUIM"
                        recomendacao = "Necessidade Urgente de Treinamento e revisão de processos operacionais básicos."
                    elif pior_classe == "serviço ruim":
                        cor = "🟡"
                        diagnostico = "SERVIÇO RUIM"
                        recomendacao = "Necessita de plano estruturado de capacitação de curto prazo para a equipe."
                    elif pior_classe == "serviço bom":
                        cor = "🟢"
                        diagnostico = "SERVIÇO BOM"
                        recomendacao = "Desempenho estável. Treinamentos de reciclagem opcionais ou de aperfeiçoamento."
                    else:
                        cor = "⭐"
                        diagnostico = "SERVIÇO EXCELENTE"
                        recomendacao = "Alta performance constatada. Compartilhar boas práticas com outros setores."

                    media_nota = registros["nota_satisfacao"].mean()
                    media_tempo = registros["tempo_resolucao_horas"].mean()
                    total_reclamacoes = registros["reclamacoes_mensais"].sum()
                    
                    relatorios_lista.append({
                        "nome": nome_servico,
                        "diagnostico": diagnostico,
                        "reclamacao": total_reclamacoes,
                        "nota": media_nota,
                        "tempo": media_tempo,
                        "cor": cor,
                        "reco": recomendacao
                    })

                for i, r in enumerate(relatorios_lista):
                    alvo_col = col_res1 if i % 2 == 0 else col_res2
                    with alvo_col:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: {'#ef4444' if 'MUITO' in r['diagnostico'] else '#f59e0b' if 'RUIM' in r['diagnostico'] else '#10b981' if 'BOM' in r['diagnostico'] else '#eab308'};">
                            <h3>{r['cor']} {r['nome'].upper()}</h3>
                            <p><b>DIAGNÓSTICO DA IA:</b> {r['diagnostico']}</p>
                            <p><b>RECOMENDAÇÃO:</b> {r['reco']}</p>
                            <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e2e8f0;"/>
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #475569;">
                                <span><b>Nota Satisfação:</b> {r['nota']:.2f}/5.0</span>
                                <span><b>Resolução Média:</b> {r['tempo']:.1f}h</span>
                                <span><b>Reclamações:</b> {r['reclamacao']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### Análise de Distribuição e Visão de Negócio")
                fig = px.scatter(
                    df, 
                    x="tempo_resolucao_horas", 
                    y="nota_satisfacao", 
                    color="resultado_ia",
                    size="reclamacoes_mensais",
                    hover_name="nome_servico",
                    labels={
                        "tempo_resolucao_horas": "Tempo de Resolução (Horas)",
                        "nota_satisfacao": "Nota de Satisfação (1 a 5)",
                        "resultado_ia": "Classificação IA",
                        "reclamacoes_mensais": "Volume de Queixas"
                    },
                    title="Mapeamento Geral: Tempo de Resposta vs Satisfação do Cliente",
                    color_discrete_map={
                        "serviço excelente": "#eab308",
                        "serviço bom": "#10b981",
                        "serviço ruim": "#f59e0b",
                        "serviço muito ruim": "#ef4444"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro durante a execução dos cálculos matemáticos da IA: {str(e)}")