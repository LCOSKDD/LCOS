from typing import List, Dict

import os, json
import asyncio

import ollama
from openai import AsyncOpenAI, OpenAI


class Backend:

    def __init__(self, model: str):
        self.model = model

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        pass
    

class OllamaBackend(Backend):

    def __init__(
        self,
        model: str = 'llama3.1',
    ):
        super().__init__(model)
    
    def __call__(
        self,
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: int = 10,
        stopping_criteria: List[str] = []
    ):
        
        stream = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature
            }
        )
        
        reply = ""
        for chunk in stream:
            reply += chunk['response']
            if any([stop in reply for stop in stopping_criteria]):
                break
        return reply
    

class OpenAIBackend(Backend):

    def __init__(self, model: str = 'o1-mini'):
        super().__init__(model)

    async def __call__(
        self, 
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
        stopping_criteria: List[str] = [],
        logprobs: bool = False,
    ):
 
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), )

        kwargs = {
            "model": self.model,
            "stream": True, 
        }
        if self.model not in ['o3-mini', 'o1-mini', 'deepseek-reasoner']:
            kwargs['temperature'] = temperature
            kwargs['max_tokens'] = max_tokens   
            kwargs['logprobs'] = logprobs

        stream = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs
        )

        reply = ""
        logprob = 0
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            # print(token)
            if token is None:
                logprob = 0 
                break
            if logprobs and chunk.choices[0].logprobs.content != []:
                logprob += chunk.choices[0].logprobs.content[0].logprob

            reply += token 
            if any([stop in reply for stop in stopping_criteria]):
                print(reply)

                break
        if logprobs:
            return reply, logprob
        return reply

            


    
    