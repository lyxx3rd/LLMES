from sentence_transformers import SentenceTransformer
import torch
import os
## 读取json文件
from tqdm import tqdm
import pandas as pd
import numpy as np
from FlagEmbedding import FlagReranker
import requests
import subprocess
import time
import threading
import requests
import time
from Classificater2 import start_flask_app

# 在单独的线程中启动Flask服务
flask_thread = threading.Thread(target=start_flask_app)
flask_thread.daemon = True  # 设置为守护线程，这样当主程序退出时，Flask服务也会停止
flask_thread.start()

## 读取文件
model_embedding = SentenceTransformer('../Model/yangjhchs/acge_text_embedding', device='cuda')
model_reranker = FlagReranker('../Model/AI-ModelScope/bge-reranker-v2-m3', use_fp16=True)

# 转换为一维的NumPy数组
def retriever_top(similarity,num_top,text_list):
    retrieve_dict = {}
    arr = np.array(similarity).flatten()
    # 找到排序后的索引
    sorted_indices = arr.argsort()
    # 因为argsort()返回的是从小到大排序的索引，所以我们需要从末尾开始取
    top3_indices = sorted_indices[-num_top:][::-1]  # 反转切片以获得最大的三个索引
    # 获取对应的值
    top3_values = arr[top3_indices]
    # 输出结果
    n = 0
    for i, v in zip(top3_indices, top3_values):
        retrieve_dict[n] = {}
        retrieve_dict[n]['Index'] = i
        retrieve_dict[n]['score'] = v
        retrieve_dict[n]['content'] = text_list[i]
        n = n+1
    return retrieve_dict

def retriever_rerank(reranker,retrieve_dict,question_intense,num_top_rerank):
    rerank_list = []
    for i in range(len(retrieve_dict)):
        rerank_list.append([str(question_intense), retrieve_dict[i]['content']])
    scores = reranker.compute_score(rerank_list, normalize=True)
    #print(scores) # [0.00027803096387751553, 0.9948403768236574]

    rerank_dict = {}
    arr = np.array(scores).flatten()
    # 找到排序后的索引
    sorted_indices = arr.argsort()
    # 因为argsort()返回的是从小到大排序的索引，所以我们需要从末尾开始取
    top3_indices = sorted_indices[-num_top_rerank:][::-1]  # 反转切片以获得最大的三个索引
    # 获取对应的值
    top3_values = arr[top3_indices]
    # 输出结果
    n = 0
    for i, v in zip(top3_indices, top3_values):
        rerank_dict[n] = {}
        rerank_dict[n]['Index'] = i
        rerank_dict[n]['score'] = v
        rerank_dict[n]['content'] = retrieve_dict[i]['content']
        n = n+1
    return rerank_dict

def retriever(model,reranker, question, Index, text_list, num_top = 10,num_top_rerank = 3):
    question_embeddings = model.encode(question, normalize_embeddings=True)
    similarity = Index @ question_embeddings.T
    retrieve_dict = retriever_top(similarity,num_top,text_list)
    rerank_dict = retriever_rerank(reranker,retrieve_dict,question,num_top_rerank)
    
    return rerank_dict

def load_message(input_content,requirements,Hierarchy,rag_content):
    if Hierarchy == "需求声明":
        messages = f"<分析报告>:{input_content}</分析报告>\n\n根据上述分析报告中给出的信息,相关的体系文件里是否有提出与‘{requirements}’对应的公司内部的要求？不需要考虑编号的一致性!请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n<符合性分析>:符合性分析\n<原文>:原文"
    elif Hierarchy == "制定流程":
        messages = f"<分析报告>:{input_content}</分析报告>\n\n根据上述分析报告中,请就‘是否直接展示了关于{requirements}的完整的管理流程或作业指导书或具体的操作方式?’,进行符合性分析.请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n\n<符合性分析>:符合性分析\n\n<原文>:原文"
    elif Hierarchy == "形成指定内容":
        messages = f"<分析报告>:{input_content}</分析报告>\n\n根据上述分析报告中,是否有直接展示了‘{requirements}’对应的要求的内容? 即该部分实现了上述要求所要求的内容,而不是提出了对应的要求,而是实现了实际的内容.请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n\n<符合性分析>:符合性分析\n\n<原文>:原文"
    elif Hierarchy == "形成程序文件":
        messages = f"<分析报告>:{input_content}</分析报告>\n\n根据上述分析报告中,是否有直接展示了‘{requirements}’对应的二级程序文件? 该二级文件应该非常详细和直接规定了上述要求对应的管理程序的各种细节,而不仅仅是一个要求.请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n\n<符合性分析>:符合性分析\n\n<文件>:文件名称及核心内容"
    elif Hierarchy == "形成其他文件":
        messages = f"<分析报告>:{input_content}</分析报告>\n\n<参考样式>{rag_content}\n\n</参考样式>根据上述分析报告中,是否有直接展示了‘{requirements}’对应的要求的所形成的文件? 我期望的内容如<参考样式>,即该部分实现了上述要求阶段所输出的结果,而不是提出了对应的要求,也不是管理流程,而是实现了实际产出的内容.请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n\n<符合性分析>:符合性分析\n\n<文件>:文件名称及核心内容"
    elif Hierarchy == "形成记录":
        messages = f"<分析报告>:{input_content}</分析报告>在上述原文中,是否有直接展示了如何将‘{requirements}’对应的要求的形成记录? 我需要形成记录的内容, 形成记录的方式以及最后记录文件的名称,如有的话再提供已经形成的记录的表格或者表头.请给出结论,符合性分析,和带文件名称的相关原文.输出格式及顺序严格遵守:<结论>:结论\n\n<符合性分析>:符合性分析\n\n<记录>:记录表格名称及核心内容"
    return messages

def call_predict_api(sentence: str, url: str = "http://127.0.0.1:5000/predict"):
    """
    调用Flask API进行预测

    :param sentence: 需要预测的句子
    :param url: API的URL，默认为本地Flask服务的URL
    :return: 预测结果或错误信息
    """
    dict_temp = {"0":"不符合","1":"符合"}
    data = {'sentence': sentence}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        result = result['prediction']
        label = dict_temp[str(result)]
        return label
    else:
        return {'error': f"Request failed with status code {response.status_code}: {response.text}"}

def check_health(url: str = "http://127.0.0.1:5000/health"):
    """
    检查Flask服务的健康状态

    :param url: 健康检查端点的URL
    :return: 服务状态
    """
    print("Flask模型启动中！")
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                status = response.json().get('status')
                if status == 'OK':
                    print("模型加载完成，服务已准备好")
                    return True
                elif status == 'Loading':
                    print("模型正在加载中...")
            else:
                print(f"健康检查失败，状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"健康检查请求失败: {e}")
        
        time.sleep(2)  # 每2秒检查一次

def pad_list(lst, target_length, padding_value=""):
    return lst + [padding_value] * (target_length - len(lst))

def QMS_review(QMS_file_name,enhaced_model_name,LLM_name, test_number = 0, Skip_repetitive = True):
    ## QMS_file_name should be "ZD","BJ","MB"
    ## enhaced_model_name should be "chat_internlm2_5","chat_qwen2_5"
    ## LLM should be "Qwen2.5","internlm2.5","Qwen","ERNIE"
    ## 检测文件
    
    
    if test_number != 0:
        print("开始测试!")
        print(f"当前QMS为{QMS_file_name},数据增强模型为{enhaced_model_name},审核模型为{LLM_name}!")
    elif test_number == 0:
        print("开始正式审核!")
        print("检测文件是否已审核")
        output_file_path = f"../output_data/output_direct_{QMS_file_name}_{enhaced_model_name}_{LLM_name}.csv"
        if os.path.exists(output_file_path):
            print("该文件已审核!")
            if Skip_repetitive:
                print("跳过该文件!")
                return None
            else:
                print("重复审核并覆盖该文件!")
        else:
            print("当前文件未审核!开始审核文件!")
            print(f"当前QMS为{QMS_file_name},数据增强模型为{enhaced_model_name},审核模型为{LLM_name}!")
    if LLM_name == "Qwen2.5":
        from LLM import chat_qwen2_5
        chat_bot = chat_qwen2_5
    elif LLM_name == "internlm2.5":
        from LLM import chat_internlm2_5
        chat_bot = chat_internlm2_5
    elif LLM_name == "Qwen":
        from LLM import chat_Qwen_Online
        chat_bot = chat_Qwen_Online
    elif LLM_name == "ERNIE":
        from LLM import chat_ERNIE_Online
        chat_bot = chat_ERNIE_Online
    elif LLM_name == "Both":
        from LLM import chat_ERNIE_Online, chat_Qwen_Online
        chat_ERNIR_bot = chat_ERNIE_Online
        chat_Qwen_bot = chat_Qwen_Online
    print("开始读取文件!")
    df_RAG = pd.read_csv(f"../RAG_data/RAG_content_file_{QMS_file_name}.csv",encoding="utf-8")
    text_list = df_RAG['content']
    Index = torch.load(f'../RAG_data/RAG_Index_file_{QMS_file_name}.pth')
    
    df_requirements = pd.read_csv(f"../RAGed_data/output_{QMS_file_name}_{enhaced_model_name}.csv",encoding="utf-8")
    
    analysis_with_knowledge_list = []
    analysis_without_knowledge_list = []
    ans_direct_kn_list = []
    result_direct_kn_list = []
    ans_direct_without_kn_list = []
    result_direct_without_kn_list = []

    temp_dict = {}
    for i in tqdm(range(len(df_requirements))):
    #for i in range(2):
        i = i
        Hierarchy = df_requirements['Hierarchy'][i]
        rag_content = df_requirements['enhanced_content'][i]
        requirements = df_requirements['requirements'][i]
        content = df_requirements['content'][i]

        #print("要求如下:\n",content)
        
        if content in temp_dict:
            analysis_with_knowledge = temp_dict[content]['analysis_with_knowledge']
            analysis_without_knowledge = temp_dict[content]['analysis_without_knowledge']
        else:
            print("生成分析！")
            temp_dict[content] = {}
            retrieve_dict = retriever(model_embedding, model_reranker,content, Index, text_list, num_top = 10,num_top_rerank = 3)
            input_content = ""
            for key, value in retrieve_dict.items():
                input_content = input_content + '\n' + retrieve_dict[key]['content']
                
            #print("\n【RAG结果如下】:\n",input_content)
            messages_with_knowledge = f"<相关原文>:\n{input_content}\n\n<\相关原文>\n\n<法律要求>:\n{content}\n\n<\法律要求>\n\n<分析层次>:\n提出需求：在原文中提出于法律表达意思相近的公司内部需求；\n制定流程：将法律要求的内容制定成对应的管理流程；\n形成指定内容：法律要求生成某个指定内容并附在文件里；\n形成程序文件：法律要求将某个管理程序形成程序文件；\n形成记录：法律要求将某个内容形成定期的记录，并保留在某个记录文件里\n\n<\分析层次>分析根据上述法律要求逐条进行分析. 对于每一条要求,适当的参考专家提出的QMS实现的六种情况‘提出需求’,‘制定流程’,‘形成其他文件’,‘形成指定内容’,‘形成程序文件’,‘形成记录’, 本公司的QMS体系文件是否满足法律要求?返回一个完整的分析报告，尽量包含更多的信息!"
            messages_without_knowledge = f"<相关原文>:\n{input_content}\n\n<\相关原文>\n\n<法律要求>:\n{content}\n\n<\法律要求>\n\n分析根据上述法律要求逐条进行分析, 本公司的QMS体系文件是否满足法律要求?返回一个完整的分析报告，尽量包含更多的信息!!"
            
            
            analysis_with_knowledge = chat_ERNIE_Online(messages_with_knowledge)
            temp_dict[content]['analysis_with_knowledge'] = analysis_with_knowledge
            analysis_without_knowledge = chat_ERNIE_Online(messages_without_knowledge)
            temp_dict[content]['analysis_without_knowledge'] = analysis_without_knowledge
            
        analysis_with_knowledge_list.append(analysis_with_knowledge)
        analysis_without_knowledge_list.append(analysis_without_knowledge)
    
            #print("\n【有知识的分析报告如下如下】:\n",analysis_with_knowledge)
            #print("\n【无知识的分析报告如下如下】:\n",analysis_without_knowledge)
        
        messages_kn = load_message(messages_with_knowledge,requirements,Hierarchy,rag_content = content)
        ans_direct_kn = chat_Qwen_Online(messages_kn)
        result_direct_kn = call_predict_api(ans_direct_kn)
        ans_direct_kn_list.append(ans_direct_kn)
        result_direct_kn_list.append(result_direct_kn)

        #print("\n【要求和层次如下】:\n",requirements,"\n",Hierarchy)
        #print("\n【有知识回答报告如下如下】:\n",ans_direct_kn)
        
        messages_without_kn = load_message(messages_without_knowledge,requirements,Hierarchy,rag_content = content)
        ans_direct_without_kn = chat_Qwen_Online(messages_without_kn)
        result_direct_without_kn = call_predict_api(ans_direct_without_kn)
        #print("\n【无知识回答报告如下如下】:\n",ans_direct_without_kn)

        ans_direct_without_kn_list.append(ans_direct_without_kn)
        result_direct_without_kn_list.append(result_direct_without_kn)

        if test_number > 0:
            if i ==test_number-1:
                df_requirements = df_requirements.head(len(analysis_with_knowledge_list))
                df_requirements['analysis_with_knowledge_list'] = analysis_with_knowledge_list
                df_requirements['analysis_without_knowledge_list'] = analysis_without_knowledge_list
                df_requirements['ans_direct_kn_list'] = ans_direct_kn_list
                df_requirements['result_direct_kn_list'] = result_direct_kn_list
                df_requirements['ans_direct_without_kn_list'] = ans_direct_without_kn_list
                df_requirements['result_direct_without_kn_list'] = result_direct_without_kn_list
                df_requirements.to_csv(f"../output_data/test_direct_output_{QMS_file_name}_{enhaced_model_name}_{LLM_name}.csv",index = False)
                print("测试结果保存成功!")
                break

    if test_number == 0:
        df_requirements = df_requirements
        df_requirements['analysis_with_knowledge_list'] = analysis_with_knowledge_list
        df_requirements['analysis_without_knowledge_list'] = analysis_without_knowledge_list
        df_requirements['ans_direct_kn_list'] = ans_direct_kn_list
        df_requirements['result_direct_kn_list'] = result_direct_kn_list
        df_requirements['ans_direct_without_kn_list'] = ans_direct_without_kn_list
        df_requirements['result_direct_without_kn_list'] = result_direct_without_kn_list
        df_requirements.to_csv(output_file_path,index = False)
        print("审核结果保存成功!")
    #return ans_expert_list, result_expert_list, result_expert_list2



    #return ans_expert_list, result_expert_list, result_expert_list2

if __name__ == '__main__':
    check_health()
    print("主程序继续运行...")
    QMS_file_name_list = ["ZD","BJ","MB"]
    enhaced_model = "chat_ERNIE"
    reviewer_model = "ERNIE"
    for QMS_file_name in QMS_file_name_list:
        QMS_review(QMS_file_name,"chat_ERNIE","Both",test_number=0,Skip_repetitive=True)

