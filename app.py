import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize the Gemma Model using Groq API
def initialize_llm(api_key):
    if not api_key.strip():
        st.error("API key is required.")
        return None
    return ChatGroq(model="Gemma-7b-It", groq_api_key=api_key)

if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip():
        st.error("Please provide the Groq API Key.")
    elif not generic_url.strip():
        st.error("Please provide the URL.")
    elif not (generic_url.startswith("http://") or generic_url.startswith("https://")):
        st.error("Please enter a valid URL. It should start with 'http://' or 'https://'.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Initialize the model
                llm = initialize_llm(groq_api_key)
                if not llm:
                    st.error("Failed to initialize the model. Please check your API key.")
                    st.stop()

                # Load the website or YT video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Ensure there are documents to summarize
                if not docs:
                    st.error("No content found at the provided URL.")
                    st.stop()

                # Chain for Summarization
                prompt_template = """
                Provide a summary of the following content in 300 words:
                Content: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                
                # Prepare the input for the chain
                input_data = {"input_documents": docs}

                # Ensure document content is valid
                combined_text = " ".join(doc.page_content for doc in docs if doc.page_content.strip())
                
                if not combined_text:
                    st.error("No valid content found in the provided documents.")
                    st.stop()

                # Run the summarization chain
                output_summary = chain.run(input_data)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
