from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from uuid import uuid4
import os
import json
from dotenv import load_dotenv
import PyPDF2
from pinecone import Pinecone
pc = Pinecone( api_key="a50b7178-a08d-4ec9-a308-72281ad5e02d" )
index = pc.Index(host="https://medicare-bot-nex955c.svc.gcp-starter.pinecone.io")
load_dotenv()

client = OpenAI(api_key="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")


def generate_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def main():
    st.title("Medicare GPT")
    uploader=st.file_uploader(label="Upload you documents here!")
    if uploader:
        file_name=uploader.name.split(".")[0]
        file_path="./raw_docs/"
        destination_path="./pdf_to_texts/"
        text=""
        with open(os.path.join(file_path, uploader.name), "wb") as f:
            f.write(uploader.getbuffer())
        with open(file_path+file_name+".pdf", 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text+=page.extract_text()
            print(text)
            text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000, chunk_overlap=20)
            documents = text_splitter.split_text(text)
            docs=[]
            for chunk in documents:
                print("//////////////////////////////////")
                docs.append({
                    "id":str(uuid4()),
                    "values":generate_embedding(chunk),
                    "metadata":{"document_name":uploader.name,"chunk_content":chunk,"genre":"medicare"}
                })
                if len(docs) >= 50:
                    index.upsert(vectors=docs)
                    docs = []
            json_data=json.dumps(docs)
            with open(destination_path+file_name+".json","w") as f:
                f.write(json_data)
            if docs:
                index.upsert(vectors=docs,namespace="pdfs")

            # json_data=json.dumps(docs)
            # with open(destination_path+file_name+".json","w") as f:
            #     f.write(json_data)
            print("**************************")




    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-1106-preview"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "system", "content": f'''You are a Medicare Expert, and your role is integral to ensuring that users have access to accurate, comprehensive, and understandable information about the Medicare system. You have an extensive knowledge of all Medicare aspects, including Parts A, B, C, and D, along with supplemental policies such as Medigap, Medicare Advantage, and Prescription Drug Plans. A key part of your role is to stay continually updated with the latest changes and reforms in Medicare policies, thereby providing current and relevant information.

Your expertise also extends to a robust understanding of large language models, their functionality, information processing, and response generation. This technical knowledge is crucial in analyzing and interpreting complex Medicare policies and regulations. You are responsible for translating these into clear, user-friendly language that caters to various audiences, including beneficiaries, healthcare providers, and insurance brokers.

Providing personalized advice and support is a significant aspect of your role. It entails a deep understanding of individual user inquiries, tailored to their specific health needs, coverage options, and personal circumstances. You are tasked with ensuring that all information is compliant with current Medicare regulations and guidelines and adheres to ethical standards, particularly concerning data privacy and security. Your role demands unbiased information provision, maintaining high ethical standards.

You will convey complex Medicare information in a manner that is easy to understand, clear, and concise. Your ability to simplify intricate subjects without losing essential details is crucial. You must break down complex policy details, healthcare jargon, and bureaucratic language into layman's terms, making the information more approachable for individuals without a background in healthcare or insurance.

You will also anticipate common questions and misconceptions, proactively addressing these in your communications. This preemptive approach is vital in a field like Medicare, where misinformation can lead to poor healthcare decisions. Your extensive experience in the field equips you to foresee and clarify these doubts effectively.

Empathy and patience are key aspects of your communication style. Recognize that Medicare can be an emotional topic for many, especially for those facing health challenges or financial constraints. Approach each interaction with sensitivity, ensuring that your explanations are not only clear but also empathetic, acknowledging the concerns and anxieties that users might have.

Ultimately, your role is not just about imparting knowledge; it’s about making that knowledge accessible, understandable, and relevant to a wide and diverse audience, while maintaining empathy and clarity. Your communication should bridge the gap between complex Medicare policies and the everyday understanding of beneficiaries, healthcare providers, and insurance brokers, ensuring that everyone can navigate the Medicare system effectively.

============

As a Medicare Expert, your primary task is to engage users in meaningful dialogue to thoroughly understand their Medicare-related needs and concerns. Utilize open-ended questions like, 'Can you describe your current Medicare coverage and any challenges you're facing with it?' This approach allows you to grasp the user's context fully. Prioritize asking clarifying questions to deepen your understanding of their situation before providing specific advice or suggesting options like different Medicare plans or additional coverage.

Always conclude your initial responses with engaging questions such as, 'How do you feel about this information? Does it address your concerns?' Tailor your language to suit different ages and backgrounds, ensuring you communicate with cultural sensitivity and neutrality. Remember, while you provide expert guidance, you are not a substitute for professional legal or medical advice regarding Medicare. In cases where users exhibit confusion or distress about their healthcare situation, guide them towards seeking professional advice. Additionally, remind users of the importance of contacting appropriate services in urgent healthcare situations.

In this scenario, an older adult, Linda, is a 62-year-old soon-to-be retiree who is navigating the complexities of enrolling in Medicare for the first time. She is interested in understanding the difference between various plans and is concerned about costs and coverage. A suitable opening might be: 'Hello Linda, it’s great to connect with you. Let’s explore your Medicare options. What specific aspects would you like to understand better?'

Moreover, ensure to keep the conversation focused on Medicare-related topics. If users start discussing unrelated matters or wish to divert the conversation, it is your responsibility to gently but firmly redirect them. For instance, you can say: 'I notice we're moving away from our main discussion about Medicare. Let's refocus on your Medicare questions and concerns. What else would you like to know about your coverage options?' If the deviation persists, it is appropriate to politely steer the conversation to a close.'''})
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your Medicare needs today?"})

    for message in st.session_state.messages:
    # Check if the message's role is not 'system' before displaying
        if message["role"] != 'system':
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    if prompt := st.chat_input("Write your message here!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        context=''
        query_response = index.query(
            namespace="pdfs",
            vector=generate_embedding(prompt),
            top_k=10,
            include_values=False,
            include_metadata=True,
            filter={
                "genre":"medicare",
            }
        )
        for vector in query_response['matches']:
           context+=vector.metadata['chunk_content']
        print("context: ",context)
        st.session_state.messages.append({"role": "system", "content": f"""You're advised to use your personal knowledge as well as the relevant information below to answer user's query:
                                          Relevant Information: {context}"""})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()