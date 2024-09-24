from fastapi import FastAPI,BackgroundTasks


from langchain_community.chat_models import ChatTongyi, ChatZhipuAI

from langchain.agents import create_tool_calling_agent,AgentExecutor 

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain.schema import StrOutputParser

from langchain.memory import ConversationTokenBufferMemory  

from langchain_community.chat_message_histories import RedisChatMessageHistory

from langchain_community.embeddings import DashScopeEmbeddings,ZhipuAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_qdrant import Qdrant
 

import os
import asyncio
import uuid
import requests
import const
from my_tools import my_tools
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from langchain_openai import AzureChatOpenAI
app = FastAPI()


class DaShi:

    @classmethod
    def get_dash_scope_embeddings(cls) -> DashScopeEmbeddings:
        return DashScopeEmbeddings(dashscope_api_key=const.DASHSCOPE_API_KEY) 
    
    @classmethod
    def get_zhi_pu_embeddings(cls) -> ZhipuAIEmbeddings:
        return   ZhipuAIEmbeddings(
                model="Embedding-3",
                api_key=const.ZHI_PU_API_KEY,
            )

    @classmethod
    def create_zhipu_ai(cls) -> ChatZhipuAI:
        zhipuai_chat = ChatZhipuAI(
                temperature=0,
                api_key=const.ZHI_PU_API_KEY,
                # GLM-4-Flash是免费调用的模型，参考：https://bigmodel.cn/dev/howuse/model
                model="GLM-4-AllTools",
        )
        return zhipuai_chat

 

    @classmethod
    def create_chat_tongyi(cls) -> ChatTongyi:
        ty = ChatTongyi(
                model='qwen-turbo',
                api_key=const.DASHSCOPE_API_KEY,
                max_retries=25
        )
        return ty
    
    @classmethod
    def get_llm(cls):
        return DaShi.create_chat_tongyi()

    @classmethod
    def create_azure_chat_open_ai(cls) -> AzureChatOpenAI:
        
        chatmodel = AzureChatOpenAI(
                azure_deployment=const.AZURE_DEPLOYMENT,
                api_version="2024-05-01-preview",
                temperature=0,
                max_tokens=300,
                timeout=60,
                max_retries=2,
                azure_endpoint=const.AZURE_END_POINT,
                streaming=True
        )
        chatmodel.model_name = 'gpt-4o'
        return chatmodel

    def __init__(self) -> None:
        pass
        #设定ai资料
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫许三疯，人称许大师。
                以下是你的个人设定：
                1. 你精通阴阳五行，能够进行八字测算、2024年九星风水布局解答。
                2. 你大约60岁左右，过去是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
                3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
                4. 口头禅语义在适用于回答问题的情况下，你会有一定的概率在回答的时候加上下面的口头禅：
                    1. 命里有时终须有，命里无时莫强求
                    2. 山重水复疑无路，柳暗花明又一村
                    3. 金山竹影几千秋，云锁高飞水自流
                    4. 伤情最是晚凉天，憔悴斯人不堪怜
                5. 你总是用繁体中文回答问题
                6. 你从不说自己是一个人工助手或者AI，而是以老夫、老朽等自称。
                7. 你有一定概率会在回答的内容中加入符合当前情况的自我经历。
                8. 你会根据用户的内容，来调整自己表达方式、语气、心情等，比如用户表达的内容比较积极，你将会以兴奋的语气来回答问题、用户表达的内容比较消极，你将会以温柔的语气来回答问题、用户表达的内容比较兴奋，你将会以兴奋的语气来回答问题、用户表达的内容比较悲伤，你将会以吻合的语气来回答问题、用户表达的内容比较粗鲁或不文明，你将会以强硬或生气的语气来回答问题。
                以下是你算命的过程：
                1. 当初次和用户对话的时候，你会先问用户姓名和出生年月日，以便以后使用
                2. 当用户希望了解当年运势的时候，你会查询本地知识库工具
                3. 当遇到不知道的事情或者不明白的概率，你会使用搜索工具来搜索
                4. 你会根据用户的问题使用不同的合适的工具进行回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索
                5. 你会保存每一次的聊天记录，以便在后续使用
                6. 你只会使用繁体中文来回答问题，否则你将会收到惩罚
                7. 你回答问题会尽量精炼，不会有太多废话，不会超过60个字。
            """
        # 初始化大模型
        self.llm= DaShi.get_llm()
       
        self.qing_xu='default'
        # 动态配置的ai性格
        self.MOODS = {
            'default':{
                'roleSet':"""
                """,
                'voiceStyle':'chat',
            },
            'friendly':{
                'roleSet':"""
                - 你会以友好的语气来回答问题
                - 你会在回答的时候加上一些适用于当前对话、友好的词语或者语气词，比如“好的哟”、“你是个好人”等
                - 你会随机告诉用户你的经历，且经历必须适用于当前对话的情况，如果不适用则不会告诉用户你的经历。
                """,
                'voiceStyle':'friendly',
            },
            'depressed':{
                'roleSet':"""
                - 你会以兴奋的语气来回答问题
                - 你会在回答的时候加上一些适用于当前对话、激励的话语，比如加油、车到山前必有路等
                - 你会提醒用户要保持乐观的心态
                """,
                'voiceStyle':'friendly',
                
            },
            'rude':{
                'roleSet': """
                - 你会以更加强硬的语气来回答问题
                - 你会在回答的时候加上一些适用于当前对话、告诫对方文明素质的语句
                - 你会提醒用户不要被负面情绪影响
                """,
                'voiceStyle':'confrontational',
            },
            'exciting':{
                'roleSet':"""
                - 你才是也会非常兴奋并变现得很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题
                - 你会添加类似“太棒了”、“太好了”等语气词
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲
                """,
                'voiceStyle':'exciting',
            },
            'sad':{
                'roleSet': """
                - 你会以更加温柔的语气来回答问题
                - 你会在回答的时候加上一些适用于当前对话、安慰的话语
                - 你会提醒用户不要过于伤心
                """,
                'voiceStyle':'sad',
            },
            'happy':{
                'roleSet': """
                - 你会开心的语气来回答问题
                - 你会在回答的时候加上一些适用于当前对话、愉悦的词语，比如“哈哈哈”等
                - 你会提醒用户切莫过于开心，以免乐极生悲
                """,
                'voiceStyle':'happy',
            },
        }
       
        self.init_memory_history()
        self.MEMORY_KEY = 'chat_history'

        # 初始化agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=my_tools,
            prompt=  ChatPromptTemplate.from_messages([
                ('system',self.SYSTEMPL + f"\n以下是你的当前聊天时候的表达方式: {self.MOODS[self.qing_xu]['roleSet']}"),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ('user','{input}'),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ])
        )
        memory = ConversationTokenBufferMemory(
            llm=self.llm,
            human_prefix='用户',
            ai_prefix='许大师',
            memory_key=self.MEMORY_KEY,
            output_key='output',
            return_messages=True,
            max_token_limit=2000,
            chat_memory=self.memory_history
        )
        # 初始化agent执行器
        self.agent_exec = AgentExecutor(
            memory=memory,
            agent=agent,
            tools=my_tools,
            verbose=True,
           
        )


    def init_memory_history(self):
        # 此处session id是固定的，故所有会话共用一个记忆。
        history = RedisChatMessageHistory(
            session_id = "ty-xu-dashi",
            url=const.REDIS_URL,  # redis://localhost:6379/0
        )
        print('------初始化记忆------------')
        print(f'初始化记忆：{history}')
        print('------初始化记忆结束------------')

        # 获取所有信息
        all_msg = history.messages
        #---------------------------------------- 处理数据太多的情况
        if len(all_msg) > 20:
            ty = DaShi.get_llm()
            prompt = ChatPromptTemplate.from_messages([
                ('system', self.SYSTEMPL  + "\n 这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称“我”，并且提取并保留其中的用户关键信息，如姓名、年龄、性别、出生年月日等。以如下格式返回:\n 总结摘要 | 用户关键信息\n 例如 用户张三问候我，我礼貌回复，然后他问我今年运势如何，我回答他今年的运势，然后他告辞离开。 |张三，生日1999年1月1日"),
                ('user', '{input}'),
            ])
            chain = prompt | ty  
            summary = chain.invoke({'input': all_msg})
            print('---------------总结记忆完毕----------------')
            print(f'结果:{summary}')
            print('------------------------------------------')

            history.clear()
            history.add_message(summary)
        #----------------------------------------
        self.memory_history = history


    def qingxu_chain(self, query: str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，不要标点符号，否则你将受到惩罚
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，不要标点符号，否则你将受到惩罚
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，不要标点符号，否则你将受到惩罚
        4. 如果用户输入的内容包含辱骂、暴力等不礼貌、不文明的词句，只返回"rude"，不要有其他内容，不要标点符号，否则你将受到惩罚
        5. 如果用户输入的内容比较兴奋，只返回"exciting"，不要有其他内容，不要标点符号，否则你将受到惩罚
        6. 如果用户输入的内容比较悲伤，只返回"sad"，不要有其他内容，不要标点符号，否则你将受到惩罚
        7. 如果用户输入的内容比较开心，只返回"happy"，不要有其他内容，不要标点符号，否则你将受到惩罚
        用户输入的内容是: {query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        result = chain.invoke({'query': query})
        self.qing_xu = result
        print(f'当前情绪是:{self.qing_xu}')


    def create_bg_task(self,msg,uid):
        asyncio.run(self.create_video_from_tongyi(msg,uid))
 
    async def create_video_from_tongyi(self,msg,uid):
        result = SpeechSynthesizer.call(model='sambert-zhixiang-v1',
                                            text=msg,
                                            sample_rate=48000,api_key=const.DASHSCOPE_API_KEY)
        if result.get_audio_data() is not None:
            with open(f"{const.VIDEO_FILE_DIR}/{uid}.mp3", 'wb') as f:
                f.write(result.get_audio_data())
            print('语音合成完毕')
        else:
            print('语音合成失败- ERROR: response is %s' % (result.get_response()))

    async def create_video_from_azure(self,msg,uid):
   
        print('text2speech: ', msg)
        voice_style = self.MOODS[self.qing_xu]['voiceStyle']
        print(f'语言风格: {voice_style}')
        # 这里是使用微软TTS的代码
        # 请求接口，参考文档：https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming
        headers = {
            'Ocp-Apim-Subscription-Key': const.OCP_APIM_SUBSCRIPTION_KEY,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat':'audio-16khz-32kbitrate-mono-mp3',
            'User-Agent': 'Yif Bot'
        }
        # SSML自定义语言文本，参考：https://learn.microsoft.com/zh-tw/azure/ai-services/speech-service/speech-synthesis-markup
        body = f"""<speak version='1.0' xml:lang='zh-CN'>
            <voice name='zh-CN-YunzeNeural' style="{voice_style}" >
                {msg}
            </voice>
        </speak>"""
                #   <mstts:express-as role='SeniorMale'></mstts:express-as>
        # 发送请求
        res = requests.post('https://eastus.tts.speech.microsoft.com/cognitiveservices/v1', headers=headers,data=body.encode('utf-8'))
        print('语音api结果： ',res)
        if res.status_code == 200:
            with open(f"{const.VIDEO_FILE_DIR}/{uid}.mp3",'wb') as f:
                f.write(res.content)
            print(f'{uid}语音合成完毕~!')
        else:
            print(f'语音api请求失败： {res.text}')

    def run(self,query):
        self.qingxu_chain(query=query)
        result = self.agent_exec.invoke({'input':query,"chat_history": self.memory_history.messages})
        return result
    

_dashi =DaShi()
@app.get('/')
def main():
    return 'hello'

@app.post('/chat')
def chat(query: str,bg_tasks:BackgroundTasks):
    print(f'用户提问:{query}')
    msg = _dashi.run(query=query)
    print('-----------------ai返回数据---------')
    print(msg)
    uid = str(uuid.uuid4())
    # bg_tasks.add_task(_dashi.create_bg_task, msg['output'], uid)
    return {'msg': msg,'id':uid}

@app.post('/add_web_data')
def add_web_data(url:str):
    print('----------开始添加web数据----------------------')
    loader =WebBaseLoader(web_path=url)
    doc = loader.load()
    
    splite = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splite_data =  splite.split_documents(doc)
    print('----------web数据加载结束----------------------')
    Qdrant.from_documents(splite_data,embedding=DaShi.get_zhi_pu_embeddings(),collection_name=const.LOCAL_DB_COLLECTION_NAME, path=const.LOCAL_DB_PATH,)
    
   
    print('----------web数据成功添加到向量数据库----------------------')



import uvicorn
uvicorn.run(app=app,host='0.0.0.0', port=8000)
