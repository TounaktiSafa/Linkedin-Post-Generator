from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

# Correct way to initialize ChatGroq
llm = ChatGroq(
    api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama3-70b-8192"
)

if __name__ == "__main__":
    response = llm.invoke("What is the most appreciated fruit in each country")
    print(response.content)