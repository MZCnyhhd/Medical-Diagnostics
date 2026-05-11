from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', base_url='http://localhost:11434', temperature=0.01)
for mode in ('invoke',):
    try:
        r = llm.invoke('请只回复OK')
        print('SYNC_OK', getattr(r,'content',str(r))[:100])
    except Exception as e:
        print('SYNC_ERR', type(e).__name__, e)
