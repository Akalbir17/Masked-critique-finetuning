from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home1/akalbirs/blurred-thoughts-SFT/results/deepscaler/2025-03-29 19_18_15/merged")
print(len(tokenizer))
