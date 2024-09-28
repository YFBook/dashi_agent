 
import gradio as gr
from server import DaShi
user_input = gr.components.Text(placeholder="请输入您想说的话",label='用户')
dashi_output = gr.components.Text(placeholder="",label='大师说')    
_dashi =DaShi()


def ai_say(user_input):
    msg = _dashi.run(query=user_input)
    return msg['output']

demo = gr.Interface(
    fn=ai_say,
    inputs=[user_input],
    title ='许大师Ai对话Demo',
    outputs=[dashi_output],
    description='输入想说的话，然后点击提交',
    clear_btn=None,
    submit_btn='告诉大师',
    allow_flagging='never',
    show_progress=True
 
)

demo.launch( )