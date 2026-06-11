from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="artifacts/best_bert_sentiment.pt",  # local path
    path_in_repo="best_bert_sentiment.pt",               # name on HF
    repo_id="Chirag238/sentimentiq",                 # your HF repo
    repo_type="model"                    
)

print("Upload complete!")
