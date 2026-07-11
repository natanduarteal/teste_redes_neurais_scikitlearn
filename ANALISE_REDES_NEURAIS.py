import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class AnalisadorServicosApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Qualidade de Serviços - IA")
        self.root.geometry("2100x1700")
        self.root.configure(bg="#f8f9fa")

        # Caminho do arquivo carregado
        self.caminho_csv = None

        # Estilização
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Treeview.Heading",
            font=("Helvetica", 10, "bold"),
            background="#dcdcdc",
        )

        # Cabeçalho Superior
        header = tk.Label(
            root,
            text="Painel de Diagnóstico de Serviços & Treinamento",
            font=("Helvetica", 14, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=12,
        )
        header.pack(fill=tk.X)

        # Container Principal
        container = tk.Frame(root, bg="#f8f9fa", padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)

        # Área de Upload
        frame_upload = tk.LabelFrame(
            container,
            text=" 1. Base de Dados ",
            font=("Helvetica", 10, "bold"),
            bg="#f8f9fa",
            padx=10,
            pady=10,
        )
        frame_upload.pack(fill=tk.X, pady=5)

        self.btn_upload = tk.Button(
            frame_upload,
            text="📂 Selecionar Arquivo CSV",
            font=("Helvetica", 10),
            bg="#34495e",
            fg="white",
            padx=10,
            command=self.fazer_upload_csv,
        )
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        self.lbl_arquivo = tk.Label(
            frame_upload,
            text="Nenhum arquivo selecionado.",
            font=("Helvetica", 10, "italic"),
            fg="#7f8c8d",
            bg="#f8f9fa",
        )
        self.lbl_arquivo.pack(side=tk.LEFT, padx=10)

        # Botão de Ação / Análise
        self.btn_analisar = tk.Button(
            container,
            text="📊 Executar Análise Preditiva (Rede Neural)",
            font=("Helvetica", 11, "bold"),
            bg="#27ae60",
            fg="white",
            pady=8,
            state=tk.DISABLED,  # Desativado até que o upload seja feito
            command=self.executar_analise,
        )
        self.btn_analisar.pack(fill=tk.X, pady=15)

        # Área de Resultados (Tabela)
        frame_tabela = tk.LabelFrame(
            container,
            text=" 2. Classificação Gerencial de Desempenho ",
            font=("Helvetica", 10, "bold"),
            bg="#f8f9fa",
            padx=5,
            pady=5,
        )
        frame_tabela.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            frame_tabela,
            columns=("Serviço", "Status", "Direcionamento Estratégico"),
            show="headings",
        )
        self.tree.heading("Serviço", text="Serviço Analisado")
        self.tree.heading("Status", text="Nível de Qualidade")
        self.tree.heading("Direcionamento Estratégico", text="Ação Recomendada")

        self.tree.column("Serviço", width=180, anchor=tk.CENTER)
        self.tree.column("Status", width=140, anchor=tk.CENTER)
        self.tree.column("Direcionamento Estratégico", width=320, anchor=tk.W)

        # Barra de rolagem para a tabela
        scrollbar = ttk.Scrollbar(
            frame_tabela, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def fazer_upload_csv(self):
        """Abre o gerenciador de arquivos para o usuário escolher o arquivo CSV."""
        arquivo_selecionado = filedialog.askopenfilename(
            title="Selecione o arquivo de Serviços",
            filetypes=[("Arquivos CSV", "*.csv")],
        )

        if arquivo_selecionado:
            self.caminho_csv = arquivo_selecionado
            nome_arquivo = os.path.basename(arquivo_selecionado)
            self.lbl_arquivo.config(
                text=f"Carregado: {nome_arquivo}",
                fg="#27ae60",
                font=("Helvetica", 10, "bold"),
            )
            self.btn_analisar.config(state=tk.NORMAL)  # Ativa o botão de análise
            messagebox.showinfo(
                "Sucesso", f"Arquivo '{nome_arquivo}' importado com sucesso!"
            )

    def simular_dados_treinamento(self):
        """Gera a inteligência base da IA (Padrões de Negócio)"""
        np.random.seed(42)
        X_treino, y_treino = [], []

        for _ in range(250):
            reclamacoes = np.random.randint(0, 50)
            tempo = np.random.uniform(0.5, 5.0)
            satisfacao = np.random.uniform(1.0, 5.0)

            # Critérios de classificação em níveis para a rede memorizar
            if reclamacoes > 35 or satisfacao < 2.0:
                nivel = 0  # Serviço Muito Ruim
            elif reclamacoes > 20 or satisfacao < 3.5:
                nivel = 1  # Serviço Ruim
            elif reclamacoes > 7 or satisfacao < 4.4:
                nivel = 2  # Serviço Bom
            else:
                nivel = 3  # Serviço Excelente

            X_treino.append([reclamacoes, tempo, satisfacao])
            y_treino.append(nivel)

        return np.array(X_treino), np.array(y_treino)

    def executar_analise(self):
        # Limpa os dados da tabela visual
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            # 1. Carrega o arquivo dinâmico que o usuário enviou
            dados_reais = pd.read_csv(self.caminho_csv)

            # Validação técnica das colunas do arquivo enviado
            colunas_obrigatorias = {
                "servico",
                "reclamacoes",
                "tempo_resolucao_horas",
                "satisfacao_cliente",
            }
            if not colunas_obrigatorias.issubset(dados_reais.columns):
                raise ValueError(
                    "O arquivo CSV precisa ter exatamente as colunas:\n"
                    "servico, reclamacoes, tempo_resolucao_horas, satisfacao_cliente"
                )

            # 2. Treinamento em tempo de execução da Rede Neural
            X_treino, y_treino = self.simular_dados_treinamento()

            scaler = StandardScaler()
            X_treino_scaled = scaler.fit_transform(X_treino)

            # Configurando o Classificador Perceptron Multicamadas (MLP)
            mlp = MLPClassifier(
                hidden_layer_sizes=(12, 12),
                max_iter=1200,
                random_state=42,
                learning_rate_init=0.01,
            )
            mlp.fit(X_treino_scaled, y_treino)

            # 3. Predição com a base carregada
            X_reais = dados_reais[
                ["reclamacoes", "tempo_resolucao_horas", "satisfacao_cliente"]
            ].values
            X_reais_scaled = scaler.transform(X_reais)

            predicoes = mlp.predict(X_reais_scaled)

            # Mapeamento do formato de resposta solicitado
            mapeamento_status = {
                0: "Serviço Muito Ruim",
                1: "Serviço Ruim",
                2: "Serviço Bom",
                3: "Serviço Excelente",
            }

            mapeamento_acao = {
                0: "⚠️ Alerta Vermelho: Necessidade crítica de treinamento e auditoria.",
                1: "🛑 Treinamento Necessário: Reciclagem operacional recomendada.",
                2: "✅ Sob Controle: Manter acompanhamento preventivo regular.",
                3: "⭐ Excelente: Padrão ouro. Utilize como modelo de benchmark.",
            }

            # 4. Inserção visual dos resultados categorizados
            for i, linha in dados_reais.iterrows():
                classe_idx = predicoes[i]
                status = mapeamento_status[classe_idx]
                acao = mapeamento_acao[classe_idx]

                self.tree.insert(
                    "", tk.END, values=(linha["servico"], status, acao)
                )

            messagebox.showinfo(
                "Processamento Concluído",
                "A inteligência artificial concluiu o mapeamento de sua operação!",
            )

        except Exception as e:
            messagebox.showerror("Erro de Processamento", f"Detalhes: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AnalisadorServicosApp(root)
    root.mainloop()