To launch the code:
```
pip install -r requirements.txt
pip install .
```
Before running the code using OpenAI models you need to get an OpenAI API Key: https://platform.openai.com/docs/overview,
and define it explicitly in you enviroment:
```
export OPENAI_API_KEY=<api_key>
```
If you are running one of the models supported by Ollama, please refer to their official documentation: https://github.com/ollama/ollama-python.
The name of the models that can be specified in input is the same provided in the official documentation by both OpenAI and Ollama. 
```
python main.py [-h] [--dataset DATASET] [--model MODEL] [--temperature TEMPERATURE] 
```
Example of prompts:
```
python main.py --dataset cancer --model gpt-4o-mini 
python main.py --dataset msu --model deepseek-r1
python main.py --dataset opioids --model mistral
```

Computing the consistency matrix might require a lot time. 
To make things easier, most of the consistency matrix for the dataset available in the folder data 
are already stored in the folder data/consistency_matrix.