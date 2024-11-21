import pandas as pd

def load_message_0(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请生成一个非常简单的关于‘{content}’的需求声明, 即这个法律要求在质量管理体系文件里对应的需求描述的示例,不能超过三句话, 不要回复其他内容."
            },
        ]
    return messages

def load_message_1(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请制定一个非常简单的关于<‘{content}’>的管理流程,执行流程或落实的示例，亦或者是作业手册的示例.只需要几条核心的流程框架,以一段话的形式,请只返回示例内容,不要说明."
            },
        ]
    return messages

def load_message_2(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请生成一个非常简单的关于‘{content}’示例内容的示例,也就是这个内容在医疗器械公司的质量管理体系文件里的具体体现一般是怎么样的?"
            },
        ]
    return messages

def load_message_3(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请生成一个非常简单的关于‘{content}’的管理程序示例,并且生成一个这个管理程序在医疗器械公司的质量管理体系文件里的具体对应的二级程序文件的文件名称."
            },
        ]
    return messages

def load_message_4(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请在医疗器械质量管理体系的框架中,生成一个非常简单的关于‘{content}’的文件示例,并且起一个文件名称."
            },
        ]
    return messages

def load_message_5(content):
    messages = [{
            "role": "user",
            "content": f"我想一个参照示例,请生成一个非常简单的关于‘{content}’的记录管理流程,并且生成一个这个管理程序在医疗器械公司的质量管理体系文件里的具体对应的需要完成的记录管表格的名称,和非常简易的表格示例."
            },
        ]
    return messages


# 加载CSV文件
df = pd.read_csv('./data/selected_requirements_v2.csv')

# 显示原始宽表
print("Original length format:")
print(len(df))

# 使用melt函数将宽表转换为窄表
df_melted = df.melt(id_vars=['documents','chapter','content','requirements'], var_name='Hierarchy', value_name='label')
df_melted = df_melted.dropna()
df_melted = df_melted[df_melted['label'] != 0]
df = df_melted.reset_index(drop=True).drop(columns=['label'])