from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='qwen2.5:7b', base_url='http://localhost:11434', temperature=0.01)
try:
    r = llm.invoke('请只回复OK')
    print('OK', str(r)[:120])
except Exception as e:
    print('ERR', type(e).__name__, e)
