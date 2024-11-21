import os
import pandas as pd
import pdfplumber
import re
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

model_embedding = SentenceTransformer('./Model/yangjhchs/acge_text_embedding', device='cuda')

def check_page(input_pdf):
    # 读取PDF文档
    with pdfplumber.open(input_pdf) as pdf:
        # 获取文档的总页数
        total_pages = len(pdf.pages)

        temp_text = ""
        # 遍历每一页
        for page_number in range(total_pages):
            # 获取当前页
            page = pdf.pages[page_number]

            # 提取文本内容
            text = page.extract_text()
            temp_text = temp_text + text
    len_text = len(temp_text)
    return total_pages, len_text


def index_count_pdf_files_in_subfolders(dir_path):
    dir_list = []
    file_name_list = []
    subfolders_list = []
    pages_num_list = []
    text_num_list = []

    # 使用os.walk遍历所有子文件夹
    for root_folder, sub_folders, files in os.walk(dir_path):
        # 遍历当前文件夹下的所有文件
        for file in files:
            if file.endswith('.pdf'):
                # 构建完整的文件路径
                file_path = os.path.join(root_folder, file)
                # 将相关信息添加到列表中
                dir_list.append(root_folder)
                file_name_list.append(file)
                subfolders_list.append(root_folder[len(dir_path):].lstrip(os.sep))  # 提取子文件夹路径
                pages_num, text_len = check_page(file_path)
                pages_num_list.append(pages_num)
                text_num_list.append(text_len)

    return dir_list, subfolders_list, file_name_list, pages_num_list, text_num_list

def split_string_if_needed(s, title, max_length=400):
    """
    根据需要将字符串分割为多个部分，每个部分长度不超过max_length。

    参数:
    s (str): 需要处理的字符串。
    title (str): 添加到每一段开头的标题。
    max_length (int, optional): 允许的最大长度。默认为400。

    返回:
    list: 包含一个或多个字符串的列表，每个字符串的长度不超过max_length，并且以title开头。
    """
    segments = []
    current_segment = ""
    current_length = 0

    # 分割字符串
    for line in s.split('\n'):
        line_length = len(line) + 1  # 加上换行符的长度

        # 检查是否添加当前行会导致超出最大长度
        if current_length + line_length > max_length:
            # 添加当前片段到segments列表，并开始新的片段
            segments.append(f"<{title}>:" + current_segment.rstrip())
            current_segment = ""
            current_length = 0

        # 如果行本身超过最大长度，尝试在行内分割
        if line_length > max_length:
            # 在行内找到一个适当的分割点
            while line:
                if len(line) <= max_length:
                    current_segment += line
                    break
                else:
                    # 寻找分割点，优先考虑空格
                    space_index = line[:max_length].rfind(' ')
                    if space_index == -1:
                        # 如果没有空格，直接截断
                        current_segment += line[:max_length-1]
                        line = line[max_length-1:]
                    else:
                        # 在空格处分割
                        current_segment += line[:space_index]
                        line = line[space_index+1:]

                # 检查是否需要添加新片段
                if len(current_segment) > 0 and len(line) > 0:
                    segments.append(f"<{title}>:" + current_segment.rstrip())
                    current_segment = ""
                    current_length = 0

        # 否则，正常添加行到当前片段
        else:
            current_segment += line + '\n'
            current_length += line_length

    # 添加最后一个片段
    if current_segment:
        segments.append(f"<{title}>:" +"\n"+ current_segment.rstrip())

    return segments

def clear_text(temp_list):
    ## 寻找"目的",从第一章开始,排除开头
    for i in range(len(temp_list)):
        id = temp_list[i].find("目的")
        if id >3:
            break

    ## 去除表头
    temp_list = temp_list[i:]
    for i in range(len(temp_list)):
        id = temp_list[i].find("页\n")
        if id <=100:
            temp_list[i] = temp_list[i][id+1:]

    ## 去除附录
    bib_list = []
    for i in reversed(range(len(temp_list))):
        if temp_list[i].startswith('\n附录'):
            bib_list.append(i)
        else:
            break
    bib_list.sort(reverse=True)
    for index in bib_list:
        temp_list.pop(index)

    ##去除开头空行
    for i in range(len(temp_list)):
        temp_list[i] = temp_list[i][1:]

    return temp_list

def extract_file_content(file_path):
    text_list = []
    with pdfplumber.open(file_path) as pdf:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]  # 获取第一页
            text = page.extract_text()
            text_list.append(text)
    return text_list

def merge_last_elements_if_short(str_list):
    # 检查列表长度是否至少为2
    if len(str_list) >= 2:
        # 获取最后一个元素
        last_element = str_list[-1]
        # 检查最后一个元素的长度是否小于200
        if len(last_element) < 100:
            # 合并最后一个元素到倒数第二个元素
            str_list[-2] += last_element
            # 删除原列表的最后一个元素
            str_list.pop()
    return str_list

def remove_line_between_chinese_chars(s):
    # 正则表达式模式，匹配两个中文字符之间的换行符
    pattern = r"([\u4e00-\u9fff])\n([\u4e00-\u9fff])"
    
    # 使用正则表达式的sub函数替换匹配到的模式
    # 第一个括号内的内容(\1)和第二个括号内的内容(\2)保持不变，只去掉换行符
    result = re.sub(pattern, r"\1\2", s)
    return result

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

def retriever(model, question, Index, text_list, num_top = 3):
    question_embeddings = model.encode(question, normalize_embeddings=True)
    similarity = Index @ question_embeddings.T
    retrieve_dict = retriever_top(similarity,num_top,text_list)
    return retrieve_dict

def embedding_QMS(file_name,embedding_test=True):
    dir_path = f'./QMS/{file_name}'  # 请将此处替换为实际文件夹路径
    print("正在处理文件,",file_name)
    #count_pdf_files_in_subfolders(dir_path)
    print("正在进行文件加载")
    dir_list,subfolders_list,file_name_list,pages_num_list,text_num_list = index_count_pdf_files_in_subfolders(dir_path)
    df = pd.DataFrame({'dir_list': dir_list,'subfolders_list':subfolders_list,'file_name_list':file_name_list,'pages_num_list':pages_num_list,'text_num_list':text_num_list})
    #df.to_csv('./data/file_index.json', index=False,encoding='utf-8')
    # 假设df是您的DataFrame
    filtered_df = df[df['text_num_list'] > 200]
    keywords_to_exclude = ["来料检验", "装配作业指导书", "软件下载作业指导书","检验规范","工艺规范","测试作业指导书","作业指引","配作业指导书","作业指导书","工艺"]
    # 创建一个正则表达式，匹配任何关键词
    pattern = '|'.join(map(re.escape, keywords_to_exclude))
    # 筛选DataFrame
    filtered_df = filtered_df[~filtered_df['file_name_list'].str.contains(pattern, regex=True)]
    print(len(list(filtered_df['file_name_list'])))

    pdf_file_content_list=[]
    pdf_file_abstract_list=[]
    text_dict = {}
    text_save_list = []
    title_list = []
    for i in tqdm(filtered_df.index, desc="Processing rows"):
        input_pdf_path = filtered_df['dir_list'][i]+'/'+filtered_df['file_name_list'][i]
        name = filtered_df['file_name_list'][i]
        temp_list = extract_file_content(input_pdf_path)
        temp_list = clear_text(temp_list)
        text_list_temp = []
        text_dict[name] = []
        title = ''.join(char for char in name if '\u4e00' <= char <= '\u9fff')
        for i in range(len(temp_list)):
            text_test = temp_list[i]
            text_test = remove_line_between_chinese_chars(text_test)
            split_list = split_string_if_needed(text_test,title)
            split_list = merge_last_elements_if_short(split_list)
            text_list_temp += split_list
        title_list = title_list + [title for _ in range(len(text_list_temp))]
        #print(f"{name}文件已切分完成,长度为:{len(text_list_temp)}")
        text_save_list = text_save_list + text_list_temp
        text_dict[name] = text_list_temp

    # 指定要保存的文件路径
    # 使用json.dump()函数将字典写入文件
    with open('./data/file_fragments.json', 'w',encoding='utf-8') as json_file:
        json.dump(text_dict, json_file,ensure_ascii=False,indent=4)
    print("切片保存成功！开始embedding!")
    with open('./data/file_fragments.json', 'r',encoding='utf-8') as f:
        data = json.load(f)
    text_save_list = []
    for key, value in data.items():
        text_save_list = text_save_list + value

    Index = model_embedding.encode(text_save_list, normalize_embeddings=True)
    torch.save(Index, f'./RAG_data/RAG_Index_file_{file_name}.pth')
    df = pd.DataFrame({"content":text_save_list})
    df.to_csv(f"./RAG_data/RAG_content_file_{file_name}.csv",index=False)
    print("embedding成功!")

    if embedding_test:
        question = "7.3.5 设计和开发评审 应评审评价设计和开发的结果满足要求的能力；"
        retrieve_dict = retriever(model_embedding, question, Index, text_list = text_save_list, num_top = 3)
        print(retrieve_dict)

if __name__ == '__main__':
    embedding_QMS('MB')
    embedding_QMS('ZD')
    embedding_QMS('BJ')