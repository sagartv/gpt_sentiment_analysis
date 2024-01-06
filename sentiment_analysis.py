import streamlit as st
import openai
from openai import OpenAI
import pandas as pd




def detect_sentiment(client, prompt, senntiments):
    system_role = f''' You are an emotionally intelligent assistant.
    Classify the sentiment of the user text by choosing ONLY ONE of the following: {sentiments}.
    After classifying the text, respond with the SENTIMENT ONLY.
    '''
    
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role':'system', 'content': system_role},
                    {'role':'user', 'content': prompt}],
        temperature = 0)
    return response.choices[0].message.content


if __name__ == "__main__":

    with open('key.txt') as f:
        api_key = f.read()
        assert api_key.startswith('sk-'), 'Error Loading the API key. The key must start with sk-'

    client = OpenAI(api_key = api_key)

    col1, col2 = st.columns([0.85,0.15])
    with col1:
        st.title('GPT Sentiment Analysis')
    with col2:
        st.image('ai.png', width = 70)
    
    with st.form(key = "sentiment_analysis_form"):
        default_sentiments = "Positive, Negative, Mixed, Angry, Despairing"
        sentiments = st.text_input("Sentiments:", value = default_sentiments)
        text = st.text_area(label = 'Enter Text to Classify:')
        submit_button = st.form_submit_button(label = "Detect Sentiment")

        if submit_button:
            detected_sentiment = detect_sentiment(client, text, sentiments)
            response = f'Sentiment Detected: {detected_sentiment} \n'
            st.write(response)

            history_row = {'Text':text, 'Detected':detected_sentiment}
            st.divider()
            if 'history' not in st.session_state:
                if response: 
                    history_df = pd.DataFrame(columns = ['Text', 'Detected'])
                    st.session_state['history'] = pd.DataFrame(history_row, index = [0])
                else:
                    st.session_state['history'] = pd.DataFrame(columns = ['Text', 'Detected'])
            else:
                history_df = st.session_state['history']
                index = len(history_df)
                new_row = pd.DataFrame(history_row, index = [index])
                st.session_state['history'] = pd.concat([history_df, new_row], axis = 0, ignore_index= True)
            
            if not st.session_state['history'].empty:
                with st.expander("History:"):
                    st.table(st.session_state['history'])

            




