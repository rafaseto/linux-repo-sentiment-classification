import requests

API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I'm shocked... what rubbish financial education did my parents give me? I'll find that out when I'm 25... I entered physics thinking that the little I would earn was 50 thousand per month, they just said it was little omg",
})
print(output)