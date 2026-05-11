import asyncio
from langchain_ollama import ChatOllama

async def main():
    llm = ChatOllama(model='qwen2.5:7b', base_url='http://localhost:11434', temperature=0.01)
    try:
        resp = await llm.ainvoke('你是分诊助手。请只回复OK。')
        print('OK:', getattr(resp, 'content', str(resp))[:200])
    except Exception as e:
        print('ERR:', type(e).__name__, e)

asyncio.run(main())
