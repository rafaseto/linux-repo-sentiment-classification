import psycopg2
from transformers import pipeline, AutoTokenizer
import torch

# Configurações do banco
connection = psycopg2.connect(
    host="localhost",
    database="ubuntu_data",
    user="postgres",
    password="postgres"
)
cursor = connection.cursor()

# Carregar o modelo e o tokenizer
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponível! Usando:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU não disponível, usando CPU.")
    
classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis", device=device)
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

# Função para verificar o comprimento do texto
def is_text_too_long(text, max_length=128):
    tokens = tokenizer.encode(text)  # Codificar o texto
    return len(tokens) > max_length  # Retorna True se o texto for maior que o limite

# Buscar textos do banco
cursor.execute("SELECT comment_id, body FROM pr_comments")
rows = cursor.fetchall()

# Processar e classificar
resultados = []
for row in rows:
    comment_id, body = row
    
    # Verificar se o texto é muito longo
    if is_text_too_long(body):
        print(f"Texto com {len(body)} caracteres excede o limite de 128 tokens. Ignorando...")
        continue  # Ignorar este comentário
    
    # Classificar o texto
    resultado = classifier(body)[0]
    resultados.append((comment_id, resultado['label'], resultado['score']))

# Atualizar resultados no banco
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS sentimento TEXT")
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS confianca NUMERIC")
for res in resultados:
    cursor.execute(
        "UPDATE pr_comments SET sentimento = %s, confianca = %s WHERE comment_id = %s",
        (res[1], res[2], res[0])
    )

connection.commit()
cursor.close()
connection.close()