from langchain.agents import tool

from langchain_community.utilities import SearchApiAPIWrapper

from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient

from langchain_community.embeddings import DashScopeEmbeddings,ZhipuAIEmbeddings

from langchain_community.chat_models.tongyi import ChatTongyi

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

import requests
import const as const
import os
 
@tool
def test():
    '''TEST TOOL'''
    return 'test'

@tool
def search_web(query:str):
    '''该工具只有在需要了解实时数据、实时信息或者不知道的事情才会使用，该工具需要接收一个字符串作为需要查询的内容'''
    print('--------------开始使用查询工具--------------')
    _tool = SearchApiAPIWrapper(searchapi_api_key=const.SEARCH_API_KYE)
    result = _tool.run(query=query)
    print('---------------实时查询完毕----------------')
    print(f'查询结果:{result}')
    print('------------------------------------------')
    return result

# 需提前导入龙年相关的知识到本地向量数据库
@tool
def get_info_from_local_db(qeury:str):
    '''该工具只有回答与九星风水布局相关问题的时候才会使用,需要传入查询的内容'''
    print('--------------开始使用本地向量数据--------------')
    client = Qdrant(
        QdrantClient(path=const.LOCAL_DB_PATH),
        const.LOCAL_DB_COLLECTION_NAME,
        # DashScopeEmbeddings(
        #         # 模型支持：https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-quick-start?spm=a2c4g.11186623.0.0.7b074c5efzzSMN
        #         model="text-embedding-v2",
        #         dashscope_api_key=const.DASHSCOPE_API_KEY,
        #     )
        ZhipuAIEmbeddings(
                model="Embedding-3",
                api_key=const.ZHI_PU_API_KEY,
            )
    )
    retriever = client.as_retriever(search_type='mmr')
    result = retriever.get_relevant_documents(query=qeury)
    print('---------------本地数据检索完毕----------------')
    print(f'结果:{result}')
    print('------------------------------------------')
    return result

@tool
def bazi_cesuan(query):
    """该工具只有做八字测算、测算八字的时候才会用到，需要传入一个dict类型的数据作为查询的入参query值，该dict数据应该包含以下key:
            - "api_key": 固定为zulZIPL1I85OlMGvk8l334Esw, 
            - "name": 用户姓名,
            - "sex": 用户性别 0男 1女 默认0,
            - "type": 历类型 0农历 1公历 默认1,
            - "year": 用户出生年 例: 1988,
            - "month": 用户出生月 例: 8,
            - "day": 用户出生日 例: 7,
            - "hours": 用户出生时 例: 12,
            - "minute":固定为0，
        其中有固定值的key（api_key和minute）无需客户提供，请自动生成，其他key的值需要客户提供，如果客户没有提供，则在以往聊天历史、记忆里面查询这些内容，如果这些内容查询不到，需要提醒客户告诉你这些内容 
    """
    url = 'https://api.yuanfenju.com/index.php/v1/Bazi/cesuan'
    print(f'需要测算的八字: {query}')
    result = requests.post(url=url, data=query)
    if result.status_code == 200:
 
        try:
            json = result.json()
            retruns_string = f"五行：{json['data']['wuxing']['detail_description']}\n财运:{json['data']['caiyun']['sanshishu_caiyun']}\n姻缘:{json['data']['yinyuan']['sanshishu_yinyuan']}"
            print('---------------八字api请求完毕----------------')
            print(f'结果:{retruns_string}')
            print('------------------------------------------')
            return retruns_string
        except Exception as e:
            print(f'测算异常: {e}')
            return '天机不可泄露'
    else:
        return '天机不可算'
    


 
my_tools = [search_web,bazi_cesuan,get_info_from_local_db]
 
 