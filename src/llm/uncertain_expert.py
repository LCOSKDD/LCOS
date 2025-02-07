import src.llm.prompts as prompts

from typing import Tuple, List

import httpx
from tqdm import tqdm
import asyncio
import numpy as np
import networkx as nx

import re

from src.llm.backend import (
    Backend,
    OllamaBackend,
    OpenAIBackend
)
from src.utils import utils

class UncertainExpert:
    def __init__(
            self,
            model: str = 'gpt-4o-mini', # 
            temperature: float = 0.0,
            verbose: int = 0,
        ) -> None:
        super().__init__()
        if ('gpt' in model or 'o1' in model):
            self.backend = OpenAIBackend(model=model)
        else:
            self.backend = OllamaBackend(model=model)
        self.model = model
        self.verbose = verbose
        self.temperature = temperature  

    async def async_query(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                response = await func(*args, **kwargs)
            except httpx.RemoteProtocolError as e:
                print(f"Attempt {attempt + 1}: RemoteProtocolError - {e}")
                asyncio.sleep(2 ** attempt)
        return response
    
    def _query(self, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    async def tripletwise(self, var_i: str, var_j: str, var_k: str) -> Tuple[Tuple[str], float]:
        prompt = prompts.prompt_triplets.format(var_i=var_i, var_j=var_j, var_k=var_k)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
        if response.find('(A)') != -1:
            return 1
        else:
            return 0
    
    async def pairwise(self, var_i: str, var_j: str) -> Tuple[Tuple[str], float]:
        reply_counter = np.zeros(3)
        for verb in prompts.causal_verbs:
            prompt = prompts.prompt_pairwise.format(var_i=var_i, verb=verb, var_j=var_j)
            kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
            if isinstance(self.backend, OpenAIBackend):
                response = await self.async_query(self.backend, prompt, **kwargs)
            else:
                response = self._query(self.backend, prompt, **kwargs)
            if response.find('(A)') != -1:
                reply_counter[0] += 1
            elif response.find('(B)') != -1:
                reply_counter[1] += 1
            else:
                reply_counter[2] += 1
            
        return (reply_counter[0] / sum(reply_counter))
    
    async def independence_test(self, i: int, j: int, k: List[int], vars: List[str]) -> float:
        var_i, var_j = vars[i], vars[j]
        vars_k = [vars[idx] for idx in k] if k else None
        vars_k = ', '.join(vars_k) if vars_k else 'nothing'
        prompt = prompts.independence_test.format(var_i=var_i, var_j=var_j, vars_k=vars_k)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
        if response.find('(A)') != -1:
            return 1
        else:
            return 0
        
    async def disambiguation(self, var_i: str, var_j: str) -> Tuple[str, str]:
        prompt = prompts.disambiguation.format(var_i=var_i, var_j=var_j)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
     
        if response.find('(A)') != -1:
            return 1
        elif response.find('(B)') != -1:
            return 0
        else:
            ValueError('No answer found')
           
    async def triplet_orientation(
            self,
            var_i: str,
            var_j: str,
            var_k: str,
            var_names: List[str],
            descriptions: List[str]
    ) -> List[Tuple]:
        var_names = list(var_names)
        descriptions = list(descriptions)
        i, j, k = var_names.index(var_i), var_names.index(var_j), var_names.index(var_k)
        description_i, description_j, description_k = descriptions[i], descriptions[j], descriptions[k]
        prompt = prompts.triplet_orientation_CoT.format(
            context=descriptions,
            var_i=var_i, 
            var_j=var_j,
            var_k=var_k,
            description_i=description_i,
            description_j=description_j,
            description_k=description_k
        )
        kwargs = {'temperature': self.temperature, 'max_tokens': 10000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)

        result = re.search('<Answer>.*?</Answer>', response.replace('\n', ''))

        result = result.group(0)
        result = result.replace('<Answer>', '')
        result = result.replace('</Answer>', '')
        result = result.replace('}', '')
        result = result.replace('{', '')
        result = result.replace('\n', '')
        result = result.replace('\t', '')
        i = -1
        while result[i] == ' ':
            i -= 1
        if result[i] != ']':
            result = result + ']'
        if result[0] != '[':
            result = '[' + result
        # eval to cast
        return eval(result)

# if __name__ == '__main__':

#     expert = UncertainExpert(variables=['smoking', 'lung cancer', 'coffee'], verbose=1)
#     print(expert.cause_effect_yes_no('smoking', 'lung cancer'))