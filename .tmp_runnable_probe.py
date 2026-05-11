import asyncio
from langchain_core.runnables import RunnableLambda

def f(x):
    return str(x)

async def af(x):
    return str(x)

try:
    r = RunnableLambda(f, afunc=af)
    print('OK', r.invoke('a'))
except Exception as e:
    print('ERR', type(e).__name__, e)
