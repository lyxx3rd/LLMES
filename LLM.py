import qianfan
from tqdm import tqdm
from openai import OpenAI

## 配置环境变量_local_model
# Set OpenAI's API key and API base to use vLLM's API server.
local_openai_api_key = "EMPTY"
local_openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=local_openai_api_key,
    base_url=local_openai_api_base,
)

## 配置环境变量_qianfan
API_Key = "osJtzljpqOGjc04LB7kvW9nA"
Secret_Key = "Hqqc1zFxaXiPMJJRSvARUQsPmZBIGbCg" 
chat_comp = qianfan.ChatCompletion(ak=API_Key, sk=Secret_Key)

## 配置环境变量_qwen
DASHSCOPE_API_KEY = "sk-632f5cf28f0a43719096801cd7c2e61a"
client_qwen = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def chat_qwen2_5(content,system_content = "你是文件整理专家，正在分析医疗器械质量管理体系的体系文件,擅长从语料库抽取信息，并组织成符合要求的文件。"):
    chat_response = client.chat.completions.create(
        model="./Qwen2___5-7B-Instruct",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content":content},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    response = chat_response.choices[0].message.content
    return response

def chat_internlm2_5(content,system_content = "你是文件整理专家，正在分析医疗器械质量管理体系的体系文件,擅长从语料库抽取信息，并组织成符合要求的文件。"):
    chat_response = client.chat.completions.create(
        model="./internlm2_5-7b-chat",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": content},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    response = chat_response.choices[0].message.content
    return response

def chat_ERNIE_Online(content,model_name = "ERNIE-4.0-turbo-8K",system_content = "你是文件整理专家，正在分析医疗器械质量管理体系的体系文件,擅长从语料库抽取信息，并组织成符合要求的文件。",top_p=0.1):
    messages = [{
            "role": "user",
            "content": content
            },
        ]
    chat_response = chat_comp.do(model=model_name, messages=messages,system = system_content,top_p = top_p)
    ans = chat_response["body"]["result"]
    return ans

def chat_Qwen_Online(content,model_name = "qwen-turbo",system_content = "你是文件整理专家，正在分析医疗器械质量管理体系的体系文件,擅长从语料库抽取信息，并组织成符合要求的文件。",top_p=0.1):
    messages=[
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': content}
        ]
    completion = client_qwen.chat.completions.create(
        model=model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        top_p=top_p,
    )
    ans = completion.choices[0].message.content
    return ans