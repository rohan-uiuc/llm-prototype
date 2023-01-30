import os

# os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HF_API_KEY'


from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
import streamlit as st
import numpy as np
import time

def process_streamlit():
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart = st.line_chart(np.random.randn(10, 2))

    for i in range(100):
        # Update progress bar.
        progress_bar.progress(i + 1)

        new_rows = np.random.randn(10, 2)

        # Update status text.
        status_text.text(
            'The latest random number is: %s' % new_rows[-1, 1])

        # Append data to the chart.
        chart.add_rows(new_rows)

        # Pretend we're doing some computation that takes time.
        time.sleep(0.1)

    status_text.text('Done!')
    st.balloons()



def process_flan_t5(prompt, question):
    # initialize HF LLM
    flan_t5 = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature":1e-10}
    )



    llm_chain = LLMChain(
        prompt=prompt,
        llm=flan_t5
    )



    return llm_chain.run(question)

    """If we'd like to ask multiple questions we can by passing a list of dictionary objects, where the dictionaries must contain the input variable set in our prompt template (`"question"`) that is mapped to the question we'd like to ask."""

    qs = [
        {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
        {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
        {'question': "Who was the 12th person on the moon?"},
        {'question': "How many eyes does a blade of grass have?"}
    ]
    res = llm_chain.generate(qs)
    res

    """It is a LLM, so we can try feeding in all questions at once:"""

    multi_template = """Answer the following questions one at a time.
    
    Questions:
    {questions}
    
    Answers:
    """
    long_prompt = PromptTemplate(
        template=multi_template,
        input_variables=["questions"]
    )

    llm_chain = LLMChain(
        prompt=long_prompt,
        llm=flan_t5
    )

    qs_str = (
        "Which NFL team won the Super Bowl in the 2010 season?\n" +
        "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
        "Who was the 12th person on the moon?" +
        "How many eyes does a blade of grass have?"
    )

    print(llm_chain.run(qs_str))


## OpenAI


# os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

def processs_openAI(prompt, question):
    davinci = OpenAI(model_name='text-davinci-003')


    llm_chain = LLMChain(
        prompt=prompt,
        llm=davinci
    )

    return llm_chain.run(question)


    qs = [
        {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
        {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
        {'question': "Who was the 12th person on the moon?"},
        {'question': "How many eyes does a blade of grass have?"}
    ]
    llm_chain.generate(qs)

    qs = [
        "Which NFL team won the Super Bowl in the 2010 season?",
        "If I am 6 ft 4 inches, how tall am I in centimeters?",
        "Who was the 12th person on the moon?",
        "How many eyes does a blade of grass have?"
    ]
    return llm_chain.run(qs)

    multi_template = """Answer the following questions one at a time.
    
    Questions:
    {questions}
    
    Answers:
    """
    long_prompt = PromptTemplate(
        template=multi_template,
        input_variables=["questions"]
    )

    llm_chain = LLMChain(
        prompt=long_prompt,
        llm=davinci
    )

    qs_str = (
        "Which NFL team won the Super Bowl in the 2010 season?\n" +
        "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
        "Who was the 12th person on the moon?" +
        "How many eyes does a blade of grass have?"
    )

    print(llm_chain.run(qs_str))

    """---"""

if __name__=='__main__':
    # build prompt template for simple question-answering
    st.title("Ask me anything Bot")
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    st.text_input("Your question", key="question", placeholder="Ask me anything")
    st.text_input("OpenAI api key", key='openai_key', placeholder="Optional")
    if st.session_state.openai_key:
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_key
    question = "Which NFL team won the Super Bowl in the 2010 season?"
    if st.session_state.question:
        question = st.session_state.question
        st.session_state.disabled = False
    else:
        st.session_state.disabled = True

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        flan_result = st.button('Flan T5', key='flan', disabled=st.session_state.disabled)
    with col2:
        open_result = st.button('Open AI', key='openai', disabled=st.session_state.disabled)




    t = st.empty()
    display_text = ''
    if flan_result:
        display_text = process_flan_t5(prompt, question)
    elif open_result:
        display_text = processs_openAI(prompt, question)
    text = ''

    for element in display_text:
        time.sleep(0.1)
        text += element
        t.text(text)

    if t != st.empty() and display_text:
        st.balloons()
