import requests
import psycopg2

API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
HEADERS = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxx"}

# Função para consultar a API do Hugging Face e obter a melhor label
def query_huggingface(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()[0]  # Pega o primeiro item da lista de resultados
        # Encontra o item com a maior pontuação
        best_label = max(result, key=lambda x: x['score'])
        print(f"Mensagem: {response.status_code}, {response.text}")
        return best_label['label'], best_label['score']
    else:
        print(f"Erro na API: {response.status_code}, {response.text}")
        return None, None
    
# Configurações do banco de dados PostgreSQL
connection = psycopg2.connect(
    host="localhost",
    database="ubuntu_data",
    user="postgres",
    password="postgres"
)
cursor = connection.cursor()

# Buscar textos do banco
cursor.execute("SELECT comment_id, body FROM pr_comments")
rows = cursor.fetchall()

# Processar textos e classificar com a API
resultados = []
for row in rows:
    id, body = row
    label, score = query_huggingface(body)
    if label and score:
        resultados.append((id, label, score))

# Atualizar os resultados no banco
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS sentimento TEXT")
cursor.execute("ALTER TABLE pr_comments ADD COLUMN IF NOT EXISTS confianca NUMERIC")

for res in resultados:
    cursor.execute(
        "UPDATE pr_comments SET sentimento = %s, confianca = %s WHERE comment_id = %s",
        (res[1], res[2], res[0])
    )

# Confirmar as alterações e fechar conexão
connection.commit()
cursor.close()
connection.close()
