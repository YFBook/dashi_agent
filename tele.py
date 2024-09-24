'''
连接TG自定义机器人
'''

import urllib.parse
import telebot
import urllib
import requests
import json
import os
import asyncio
from const import VIDEO_FILE_DIR,TG_BOT_KEY
bot = telebot.TeleBot(TG_BOT_KEY)
AI_SERVER_URL = os.getenv('AI_SERVER_URL','http://localhost:8000')
@bot.message_handler(commands=['start'])
def start_message(message):
    # 引用message进行回复
    # bot.reply_to(message,'你好！我是许大师，有什么能帮助你')
    
    # 不引用上文，直接回复
    bot.send_message(message.chat.id, '你好！我是许大师，有什么能帮助你')


# 接收任意消息文本时候的回调
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    try:
        encoded_text = urllib.parse.quote(message.text)
        response = requests.post(f'{AI_SERVER_URL}/chat?query=' + encoded_text,timeout=300)
        if response.status_code == 200:
            ai_say = json.loads(response.text)
            if "msg" in ai_say:
                bot.reply_to(message,ai_say['msg']['output'].encode('utf-8'))
                if 'id' in ai_say:              
                    audio_path = f"{VIDEO_FILE_DIR}/{ai_say['id']}.mp3"
                    asyncio.run(check_audio(message,audio_path))
            else:
                bot.reply_to(message,'对不起，天机不可泄露')
            

    except requests.RequestException as err:
        print(f'请求异常:{err}')
        bot.reply_to(message,'对不起，天机不可泄露')
    except Exception as err:
        print(f'服务异常:{err}')
        bot.send_message(message.chat.id,'我的大脑出事了，赶紧联系yif') 
    
async def check_audio(message, audio_path):
    check_times = 0
    while True:
        try:
            if os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    # bot.send_audio(message.chat.id,f)
                    bot.send_voice(message.chat.id,f)
                os.remove(audio_path)
                break
            else:
                print('等待获取音频')
                await asyncio.sleep(1)
                check_times += 1
            if check_times > 10:
                print('超过10秒未获取音频')
                break    
        except Exception as err:
            print(f'获取音频失败:{err}')
            break

bot.infinity_polling()
