import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load tickets
def load_tickets(csv_path="data/support_tickets.csv"):
    return pd.read_csv(csv_path)

# Zero-shot tagging
def zero_shot_tag(ticket_text):
    prompt = PromptTemplate(
        input_variables=["ticket"],
        template="Classify this support ticket into top 3 categories:\n{ticket}"
    )
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(ticket=ticket_text)
