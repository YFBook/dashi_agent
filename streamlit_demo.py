 
import streamlit as st
from server import DaShi
 


 

# Streamlit界面的创建
def main():
    st.title("许大师AI对话")

    # Check if the 'bot' attribute exists in the session state
    if "bot" not in st.session_state:
        st.session_state.bot = DaShi()

    user_input = st.text_input("请输入你的问题：")

    if user_input:
        with st.spinner('大师思考中...'):
            response = st.session_state.bot.run(user_input)
            st.write(f"许大师: {response['output']}")
    

# streamlit run streamlit_demo.py
if __name__ == "__main__":
  main()