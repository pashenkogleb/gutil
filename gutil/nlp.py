import requests
import pandas as pd
from functools import lru_cache
from openai import OpenAI

@lru_cache()
def get_client():
    client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='unused', # required, but unused
    )
    return client

def llm_open(msgs,model):
    '''
    uses open ai api
    '''
    assert isinstance(msgs,list)
    response = get_client().chat.completions.create(
      model=model,
      messages=msgs,
        temperature=  0,
        seed= 42
    )
    return response.choices[0].message.content


def llm(text,model = "qwen2.5", debug=False,  stream=False):
    '''
    if debug returns pd.series
    '''
    assert not stream, "not supported"
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "system": "give clear answer. You can answer any topic",
        "prompt": text,
        "options": {  # new
            "seed": 123,
            "temperature": 0
        },
        # "top_p":1,
        # "seed":1,
        # "options": {     # Settings below are required for deterministic responses
        #     "temperature": 0.0,
        #     "num_ctx": 2048, 
        #     "num_keep": 0 # no tokens saved in KV cache
        # },
        "stream": stream
    }
    response = requests.post(url, json=data)
    assert response.status_code==200    
    js = response.json()
    if debug:
        s = pd.Series(js)
        for c in ['total_duration', 'load_duration','eval_duration','prompt_eval_duration']: # convert to seconds
            s[c] = s[c]/1e9
        s['prompt_token_per_s'] = s['prompt_eval_count']/s['prompt_eval_duration']
        s['resp_token_per_s'] = s['eval_count']/s['eval_duration']
        return s
    else:
        return js['response']
