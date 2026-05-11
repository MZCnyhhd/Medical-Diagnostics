from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='qwen2.5:7b', base_url='http://localhost:11434/v1', api_key='ollama', temperature=0.01)
try:
    r = llm.invoke('请只回复OK')
    print('OK', getattr(r,'content',str(r))[:120])
except Exception as e:
    print('ERR', type(e).__name__, e)
