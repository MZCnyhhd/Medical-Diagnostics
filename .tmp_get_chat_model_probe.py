import asyncio
from src.services.llm import get_chat_model

async def main():
    m = get_chat_model(override_provider='ollama')
    try:
        r = await m.ainvoke('请只回复OK')
        print('OK', str(r)[:120])
    except Exception as e:
        print('ERR', type(e).__name__, e)

asyncio.run(main())
