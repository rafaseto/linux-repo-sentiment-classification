import psycopg2  # Importa o módulo para interação com o banco de dados PostgreSQL
from transformers import pipeline, AutoTokenizer  # Importa o pipeline e o AutoTokenizer da biblioteca transformers para trabalhar com modelos de NLP
import torch  # Importa o módulo PyTorch para trabalhar com GPU/CPU e redes neurais

# Configurações do banco de dados
connection = psycopg2.connect(  # Conecta-se ao banco de dados PostgreSQL
    host="localhost",  # Endereço do servidor do banco (localhost significa que é local)
    database="ubuntu_data",  # Nome do banco de dados que será acessado
    user="postgres",  # Nome de usuário para autenticação no banco de dados
    password="postgres"  # Senha para autenticação
)
cursor = connection.cursor()  # Cria um cursor para executar comandos no banco de dados

# Carregar o modelo e o tokenizer
if torch.cuda.is_available():  # Verifica se há uma GPU disponível
    device = torch.device("cuda")  # Se houver, usa a GPU
    print("GPU disponível! Usando:", torch.cuda.get_device_name(0))  # Exibe o nome da GPU
else:
    device = torch.device("cpu")  # Se não houver GPU, usa o CPU
    print("GPU não disponível, usando CPU.")  # Informa que o CPU será usado

# Carregar o modelo e tokenizer do HuggingFace para análise de sentimentos
classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis", device=device)  
# O pipeline é uma abstração de alto nível para execução de tarefas de NLP. Aqui, é usado para classificação de sentimentos.
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")  
# Carrega o tokenizer que foi treinado para o modelo específico "finiteautomata/bertweet-base-sentiment-analysis"

# Função para verificar o comprimento do texto
def is_text_too_long(text, max_length=128):  
    # Função que verifica se o texto excede o comprimento máximo permitido (em tokens)
    tokens = tokenizer.encode(text)  # Codifica o texto em tokens
    return len(tokens) > max_length  # Retorna True se o texto for maior que o limite especificado

# Buscar textos do banco de dados
cursor.execute("SELECT comment_id, body FROM pr_comments")  
# Executa a consulta SQL para selecionar o 'comment_id' e o 'body' (corpo do comentário) da tabela 'pr_comments'
rows = cursor.fetchall()  # Recupera todas as linhas resultantes da consulta

# Processar e classificar
resultados = []  # Lista onde os resultados da classificação serão armazenados
for row in rows:  # Itera sobre as linhas retornadas
    comment_id, body = row  # Extrai o 'comment_id' e o 'body' de cada linha
    
    # Verificar se o texto é muito longo
    if is_text_too_long(body):  # Verifica se o corpo do comentário excede o limite de tokens
        print(f"Texto com {len(body)} caracteres excede o limite de 128 tokens. Ignorando...")  # Informa que o comentário foi ignorado
        continue  # Ignora o processamento deste comentário e passa para o próximo
    
    # Classificar o texto
    resultado = classifier(body)[0]  # Classifica o sentimento do comentário (retorna um dicionário com 'label' e 'score')
    resultados.append((comment_id, resultado['label'], resultado['score']))  # Armazena o 'comment_id', o 'label' e o 'score' na lista de resultados

# Atualizar resultados no banco de dados
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS sentimento TEXT")  # Adiciona a coluna 'sentimento' na tabela se ela não existir
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS confianca NUMERIC")  # Adiciona a coluna 'confianca' para armazenar a confiança da classificação
for res in resultados:  # Itera sobre os resultados da classificação
    cursor.execute(  # Executa um comando SQL para atualizar o banco de dados
        "UPDATE pr_comments SET sentimento = %s, confianca = %s WHERE comment_id = %s",  # Atualiza as colunas 'sentimento' e 'confianca' com os valores dos resultados
        (res[1], res[2], res[0])  # Passa os valores do 'label', 'score' e 'comment_id' para a consulta
    )

connection.commit()  # Aplica as alterações no banco de dados
cursor.close()  # Fecha o cursor
connection.close()  # Fecha a conexão com o banco de dados