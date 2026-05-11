import os, traceback, asyncio
os.environ['LLM_PROVIDER']='local'
os.environ['LOCAL_MODEL_PATH']=r'E:\LocalModel\LLM\Med\HuatuoGPT-7B'
from src.services.llm import get_chat_model

async def main():
    m = get_chat_model(override_provider='local')
    try:
        r = await m.ainvoke('请只回复OK')
        print('OK', str(getattr(r,'content',r))[:200])
    except Exception as e:
        print('ERR', type(e).__name__, e)
        traceback.print_exc()

asyncio.run(main())
