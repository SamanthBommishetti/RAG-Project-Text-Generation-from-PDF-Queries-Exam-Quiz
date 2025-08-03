import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from fpdf import FPDF
import time
import random
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

huggingface_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "area" not in st.session_state:
    st.session_state.area = ""

if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = {"mcq": [], "fill_in_blank": []}

if "question_paper" not in st.session_state:
    st.session_state.question_paper = {"short_questions": [], "long_questions": []}  
    st.session_state.questions_generated = 0

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a page:", ["Home", "Exam", "Question Paper"])    

st.sidebar.title("                                        ")
st.sidebar.title("                                        ")
st.sidebar.title("                                        ")

st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()

        with open("temp_uploaded_file.pdf", "wb") as f:  
            f.write(uploaded_file.getbuffer())

        st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

##### groq models #####
#llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-8b-8192-tool-use-preview")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
#     temperature = 0.5
# )

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)





class PDF(FPDF):
    def header(self):
        # Custom header code if needed
        pass

    def footer(self):
        # Custom footer code if needed
        pass

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()




if option == "Home":
    st.title("Text Generation with ChatGroq")

    st.markdown("""
        <style>
        .chat-container {
            height: 350px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #2E2E2E; 
            color: #fff; 
        }
                
        .big-question {
            font-size: 18px;
            font-weight: bold;
            color: #fff; 
        }
                
        .chat-container hr {
            border: 0.5px solid #ccc;
        }
                
        .stButton button {
            background-color: red;
            color: white;
            width: 100%;
            height: 50px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Display chat history in a scrollable container
    if st.session_state.chat_history:
        chat_display = ""
        for chat in st.session_state.chat_history:
            chat_display += f"<div class='big-question'>Question :- {chat['question']}</div>\n"
            chat_display += f"<div>Answer :- \n\n{chat['answer']}</div>\n"
            chat_display += "<hr>\n"
        
        st.markdown(f"<div class='chat-container'>{chat_display}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-container'>No chat history yet.</div>", unsafe_allow_html=True)

    def update_chat():
        input_text = st.session_state.input_text
        if input_text:
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            #start_time = time.time()
            response = retrieval_chain.invoke({"input": input_text})
            #end_time = time.time()

            st.session_state.chat_history.append({"question": input_text, "answer": response["answer"]})

            # total_time = end_time - start_time
            # st.write(f"Response time: {total_time:.2f} seconds")

            # Clear the input text
            st.session_state.input_text = ""

    with st.form(key='chat_form', clear_on_submit=True):
        st.text_input("Input your question", key="input_text")
        submit = st.form_submit_button(label="Send", on_click=update_chat)








elif option == "Exam":
    st.title("Exam Question Generation")

    num_questions = st.selectbox("Select number of questions to generate:", [5, 10, 15, 20], index=0)

    question_type = st.selectbox("Select the type of questions to answer:", ["All Questions", "Multiple Choice Questions", "Fill-in-the-Blank Only"], index=0)

    if uploaded_file is not None:

        if 'questions_generated' in st.session_state and st.session_state.questions_generated != num_questions:
            st.session_state.exam_questions = {"mcq": [], "fill_in_blank": [], "mcq_answers": [], "fill_in_blank_answers": []}
            st.session_state.questions_generated = num_questions

        mcq_prompt = f"""
        Generate {num_questions} multiple-choice questions based on the following context.
        Each question should have 4 options. Format the questions as "Q1. <Question>", followed by options "A) <Option 1>", "B) <Option 2>", "C) <Option 3>", "D) <Option 4>".
        Please Please Remember to provide perfect and suitable option where the option should not repeated more then once for entire questions.
        <context>
        {{context}}
        <context>
        """

        fill_in_blank_prompt = f"""
        Generate {num_questions} fill-in-the-blank questions based on the following context.
        Each question should contain a blank ("___") and provide the correct answer for each question in the format "Answer: <correct_answer>" after the question.
        Remember to provide perfect and suitable answers where the answer should not repeated more then once.
        <context>
        {{context}}
        <context>
        """

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Number of questions selected: ", num_questions)
        with col2:
            if st.button("Prepare Questions"):
                # Generate questions
                def generate_questions():
                    mcq_questions_formatted = []
                    fill_in_blank_questions_filtered = []
                    mcq_answers = []
                    fill_in_blank_answers = []

                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    for i, chunk in enumerate(st.session_state.documents):
                        context_text = chunk.page_content

                        # Generate multiple-choice questions
                        if len(mcq_questions_formatted) < num_questions:
                            mcq_response = retrieval_chain.invoke({"input": mcq_prompt.format(context=context_text)})
                            mcq_questions = mcq_response["answer"]

                            current_question = ""
                            options = []
                            mcq_answers = []
                            for line in mcq_questions.split("\n"):
                                if line.startswith("Q"):
                                    if current_question:
                                        random.shuffle(options)

                                        for j, option in enumerate(options):
                                            if option.endswith(mcq_answers[-1]):
                                                mcq_answers[-1] = chr(65 + j)  
                                                break
                                        mcq_questions_formatted.append(current_question.strip() + "\n" + "\n".join(options))

                                    current_question = f"**{line.strip()}**\n\n"
                                    options = []
                                elif line.startswith(("A)", "B)", "C)", "D)")):
                                    options.append(f"- {line.strip()}")
                                    if line.startswith("A)"):
                                        mcq_answers.append(line[3:].strip())

                            # Handle the last question
                            if current_question and len(mcq_questions_formatted) < num_questions:
                                if options:
                                    random.shuffle(options)

                                    for j, option in enumerate(options):
                                        if option.endswith(mcq_answers[-1]):
                                            mcq_answers[-1] = chr(65 + j)  
                                            break

                                    mcq_questions_formatted.append(current_question.strip() + "\n" + "\n".join(options))

                        # Generate fill-in-the-blank questions
                        if len(fill_in_blank_questions_filtered) < num_questions:
                            fill_in_blank_response = retrieval_chain.invoke({"input": fill_in_blank_prompt.format(context=context_text)})
                            fill_in_blank_questions = fill_in_blank_response["answer"].split("\n")

                            for line in fill_in_blank_questions:
                                if "Answer:" in line:
                                    question_part = line.split("Answer:")[0].strip()
                                    answer_part = line.split("Answer:")[1].strip()
                                    
                                    question_part = question_part.rstrip("(")  

                                    answer_part = answer_part.rstrip(").")  
                                    
                                    if "___" in question_part:
                                        fill_in_blank_questions_filtered.append(question_part)
                                        fill_in_blank_answers.append(answer_part)

                        # Stop if we have enough questions
                        if len(mcq_questions_formatted) >= num_questions and len(fill_in_blank_questions_filtered) >= num_questions:
                            break

                    while len(mcq_questions_formatted) < num_questions:
                        mcq_questions_formatted.append("MCQ Placeholder Question")
                        mcq_answers.append("Unknown")

                    while len(fill_in_blank_questions_filtered) < num_questions:
                        fill_in_blank_questions_filtered.append("Fill-in-the-Blank Placeholder Question")
                        fill_in_blank_answers.append("Unknown")

                    st.session_state.exam_questions["mcq"] = mcq_questions_formatted
                    st.session_state.exam_questions["fill_in_blank"] = fill_in_blank_questions_filtered
                    st.session_state.exam_questions["mcq_answers"] = mcq_answers
                    st.session_state.exam_questions["fill_in_blank_answers"] = fill_in_blank_answers

                generate_questions()
                st.session_state.questions_generated = num_questions

        # Display exam questions
        user_answers = {}

        if question_type == "All Questions":
            st.subheader("Multiple Choice Questions")
            for i, question in enumerate(st.session_state.exam_questions["mcq"], 1):
                if question.strip():  
                    st.markdown(question)
                    user_answers[f"mcq_{i}"] = st.selectbox(f" ", [f"Select the Option Q{i}", "A", "B", "C", "D"], key=f"mcq_{i}")

            st.subheader("Fill-in-the-Blank Questions")
            for i, question in enumerate(st.session_state.exam_questions["fill_in_blank"], 1):
                if question.strip() and "___" in question:  
                    st.markdown(question)
                    
                    user_answers[f"fib_{i}"] = st.text_input(f"Enter your answer here", key=f"fib_{i}")

        if question_type == "Multiple Choice Questions":
            st.subheader("Multiple Choice Questions")
            for i, question in enumerate(st.session_state.exam_questions["mcq"], 1):
                if question.strip():  
                    st.markdown(question)
                    user_answers[f"mcq_{i}"] = st.selectbox(f" ", [f"Select the Option Q{i}", "A", "B", "C", "D"], key=f"mcq_{i}")

        if question_type == "Fill-in-the-Blank Only":
            st.subheader("Fill-in-the-Blank Questions")
            for i, question in enumerate(st.session_state.exam_questions["fill_in_blank"], 1):
                if question.strip() and "___" in question:  
                    st.markdown(question)
                    
                    user_answers[f"fib_{i}"] = st.text_input(f"Enter your answer here", key=f"fib_{i}")

        # Submit button for answers
        if st.button("Submit Answers"):
            st.subheader("Results")

            max_questions = num_questions

            # Ensure similar handling for MCQs
            if question_type == "All Questions" or question_type == "Multiple Choice Questions":
                st.subheader("Multiple Choice Questions Results")
                for i, correct_answer in enumerate(st.session_state.exam_questions["mcq_answers"], 1):
                    if i > max_questions:
                        break  

                    user_answer = user_answers.get(f"mcq_{i}", "").strip()  
                    if user_answer == correct_answer:
                        st.markdown(f"**Q{i}:** Correct! ✅")
                    else:
                        st.markdown(f"**Q{i}:** Wrong ❌ (Correct answer: {correct_answer})")            
            
            # Check Fill-in-the-Blank answers
            if question_type == "All Questions" or question_type == "Fill-in-the-Blank Only":
                st.subheader("Fill-in-the-Blank Results")
                for i, correct_answer in enumerate(st.session_state.exam_questions["fill_in_blank_answers"], 1):
                    if i > max_questions:
                        break  

                    user_answer = user_answers.get(f"fib_{i}", "").strip()
                    if user_answer.lower() == correct_answer.lower():
                        st.markdown(f"**Q{i}:** Correct! ✅")
                    else:
                        st.markdown(f"**Q{i}:** Wrong ❌ (Correct answer: {correct_answer})")








if option == "Question Paper":
    st.title("Question Paper Generation")

    col0, col01 = st.columns([3,1])  
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # User inputs for customizing the content
    with col0:
        school_name = st.text_input("Enter School Name", " Bishop Beretta HIGH SCHOOL")
    with col01:    
        class_info = st.text_input("Enter Class", "VI")

    with col1:
        test_type = st.text_input("Enter Test Type", "UNIT TEST I")
    with col2:
        subject_info = st.text_input("Enter Subject", "ENGLISH")
    with col3:
        time_info = st.text_input("Enter Time", "2.10 HOURS")
    with col4:
        total_marks = st.text_input("Enter Total Marks", "70")
                       
    
    col21, col22 = st.columns([2,1])
    col23, col24 = st.columns([2,1])
    
    with col21:
        condition_info_a = st.text_input("Section - A ", "i: Answer all the questions")
    with col22:
        marks_a = st.text_input(' ', "10 X 2 = 20")

    with col23:
        condition_info_b = st.text_input("Section - B ", "i: Answer any (FIVE) questions ")
    with col24:
        marks_b = st.text_input(' ', "5 X 10 = 50")

    # A4 paper style
    st.markdown("""
        <style>
        .a4-paper1 {
            width: 794px;
            height: 1123px;
            margin: 0 auto;
            padding: 40px;
            border: 1px solid #000;
            background-color: white;
            font-family: 'Arial', sans-serif;
            position: relative;
        }
        .a4-paper2 {
            width: 794px;
            height: 1123px;
            margin: 0 auto;
            padding: 40px;
            border: 1px solid #000;
            background-color: white;
            font-family: 'Arial', sans-serif;
            position: relative;
        }
        .heading {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: black;
        }
        .subheading {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: black;
        }
        .info {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: black;
        }
        .line {
            border: none;
            border-top: 2px solid black;
            width: 100%;
        }
        .onlynote {
            font-size: 16px;
            text-align: left;
            font-weight: bold;
            color: black;
                
        }
        .note {
            font-size: 16px;
            text-align: left;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)    

    col5, col6 = st.columns([1, 1])
    with col5:
        num_questions_short = st.selectbox("Select number of short questions (2-marks) to generate:", [5, 10, 15], index=0)
    with col6:
        num_questions_long = st.selectbox("Select number of long questions (10-marks) to generate:", [5, 8, 10, 12], index=0)

    final_questions_a = []
    final_questions_b = []

    if uploaded_file is not None:

        if st.session_state.questions_generated != (num_questions_short, num_questions_long):
            st.session_state.question_paper = {"short_questions": [], "long_questions": []}
            st.session_state.questions_generated = (num_questions_short, num_questions_long)


        section_a_prompt = f"""
        Generate {num_questions_short} 2-mark questions based on the following context.
        Each question should be concise and focused, asking for factual, definitions, or short descriptive answers.
        Use question types like what, how, which, define, or differentiate. 
        Do not provide any answers or explanations. Only generate the questions.
        <context>
        {{context}}
        <context>
        """

        section_b_prompt = f"""
        Generate {num_questions_long} 10-mark questions based on the following context.
        The questions should require detailed explanations, in-depth analysis, and assess comprehensive understanding of concepts.
        Use question types like explain, discuss, analyze, and evaluate.
        Do not provide any answers or explanations. Only generate the questions.
        <context>
        {{context}}
        <context>
        """

        col7, col8 = st.columns([2, 1])

        with col8:
            if st.button("Prepare Questions"):
                def generate_questions():
                    short_questions = []
                    long_questions = []

                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    for chunk in st.session_state.documents:
                        context_text = chunk.page_content

                        # Generate short questions
                        if len(short_questions) < num_questions_short:
                            short_response = retrieval_chain.invoke({"input": section_a_prompt.format(context=context_text)})
                            #st.write("Short response received:", short_response)

                            if "answer" in short_response and short_response["answer"].strip():
                                response_text = short_response["answer"]
                                question_lines = response_text.split("\n")

                                for line in question_lines:
                                    line = line.strip()
                                    if not line or line.startswith("Please note") or line.startswith("Here are") or line.startswith("Note"):
                                        continue
                                    short_questions.append(line)                                

                        # Generate long questions
                        if len(long_questions) < num_questions_long:
                            long_response = retrieval_chain.invoke({"input": section_b_prompt.format(context=context_text)})
                            #st.write("Short response received:", long_response)

                            if "answer" in long_response and long_response["answer"].strip():
                                response_text = long_response["answer"]
                                question_lines = response_text.split("\n")

                                for line in question_lines:
                                    line = line.strip()
                                    if not line or line.startswith("Please note") or line.startswith("Here are") or line.startswith("Note"):
                                        continue
                                    long_questions.append(line)

                        # Stop if we have enough questions
                        if len(short_questions) >= num_questions_short and len(long_questions) >= num_questions_long:
                            break

                    # Fill remaining questions with placeholders
                    while len(short_questions) < num_questions_short:
                        short_questions.append("Placeholder question for Section A.")
                    while len(long_questions) < num_questions_long:
                        long_questions.append("Placeholder question for Section B.")

                    st.session_state.question_paper["short_questions"] = short_questions
                    st.session_state.question_paper["long_questions"] = long_questions

                generate_questions()
                st.session_state.questions_generated = (num_questions_short, num_questions_long)


        if st.session_state.question_paper["short_questions"]:
            for question in st.session_state.question_paper["short_questions"]:
                final_questions_a.append(question)
        else:
            st.write("Please Select number of questions to generate and Click Prepare Questions &nbsp;&nbsp; 👆")

        if st.session_state.question_paper["long_questions"]:
            for question in st.session_state.question_paper["long_questions"]:
                final_questions_b.append(question)


    section_a_questions = ''.join([f"<div class='note'>{question}<br><br></div>" for question in final_questions_a])

    # Generate the A4 paper content
    section_a_content = f'''
    <div class="a4-paper1">
        <div class="heading">{school_name}</div>
        <div class="subheading">{test_type}</div>
        <div class="info">CLASS : {class_info} | SUBJECT : {subject_info}</div>
        <div class="info">TIME: {time_info} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; TOTAL MARKS: {total_marks}</div>
        <div class="line">__________________________________________________________________</div>
        <div class="onlynote">Note :</div>
        <div class="note">i: The question paper has been divided into two sections A and B.<br>                                   
                          ii: 10 minutes of time is allowed exclusively for reading the question paper and Remaining time for writing the answer's.<br>
                          iii: Answer all the questions on a separate answer booklet supplied to you.</div>
        <div class="subheading">SECTION : A</div>
        <div class="note">{condition_info_a} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {marks_a}</div>
        <div class="line">__________________________________________________________________</div>
        {section_a_questions}
    </div>
    '''

    section_b_questions = ''.join([f"<div class='note'>{question}<br><br></div>" for question in final_questions_b])

    # Generate the A4 paper content
    section_b_content = f'''
    <div class="a4-paper2">
        <div class="subheading">SECTION : B</div>
        <div class="note">{condition_info_b} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {marks_b}</div>
        <div class="line">__________________________________________________________________</div>
        {section_b_questions}
    </div>
    '''

    st.markdown(section_a_content, unsafe_allow_html=True)
    st.markdown(section_b_content, unsafe_allow_html=True)

    if st.button("Prepare To Download PDF"):
        pdf = PDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 10, school_name, ln=True, align="C")
        pdf.set_font("Arial", "I", 20)
        pdf.cell(0, 10, test_type, ln=True, align="C")
        pdf.set_font("Arial", "I", 16)
        pdf.cell(0, 10, f"CLASS : {class_info} | SUBJECT : {subject_info}", ln=True, align="C")
        pdf.cell(0, 10, f"TIME: {time_info}                                                            TOTAL MARKS: {total_marks}", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        line_y = pdf.get_y()  
        pdf.line(10, line_y, 200, line_y)        

        # Note section
        pdf.set_font("Arial", "", 16)
        pdf.cell(0, 10, "Note:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, "i: The question paper has been divided into two sections A and B.\nii: 10 minutes of time is allowed exclusively for reading the question paper and Remaining time for writing the answer's.\niii: Answer all the questions on a separate answer booklet supplied to you.", align="L")

        # Section A
        pdf.set_font("Arial", "I", 16)
        pdf.cell(0, 10, "                                                " "SECTION : A", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"{condition_info_a}                                                                                                       {marks_a}", ln=True, align="C")
        line_y = pdf.get_y()  
        pdf.line(10, line_y, 200, line_y)

        # short questions 
        for question in final_questions_a:
            pdf.set_font("Arial", "", 12)
            cell_height = pdf.font_size * 1.5  
            wrapped_text = pdf.multi_cell(0, cell_height, question, align="L", split_only=True)

            for line in wrapped_text:
                if pdf.get_y() + cell_height > 270:  
                    pdf.add_page()
                pdf.cell(0, cell_height, line, ln=True)

            pdf.ln(5)
        
        pdf.add_page()  

        # Section B
        pdf.set_font("Arial", "I", 16)
        pdf.cell(0, 10, "                                                " "SECTION : B", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"{condition_info_b}                                                                                                {marks_b}", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        line_y = pdf.get_y()  
        pdf.line(10, line_y, 200, line_y)

        # long questions
        for question in final_questions_b:
            pdf.set_font("Arial", "", 12)
            cell_height = pdf.font_size * 1.5  
            wrapped_text = pdf.multi_cell(0, cell_height, question, align="L", split_only=True)

            for line in wrapped_text:
                if pdf.get_y() + cell_height > 270:  
                    pdf.add_page()
                pdf.cell(0, cell_height, line, ln=True)

            pdf.ln(5)

        # Save the PDF
        pdf.output("question_paper.pdf")

        # Offer the PDF for download
        with open("question_paper.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="question_paper.pdf", mime="application/pdf")    