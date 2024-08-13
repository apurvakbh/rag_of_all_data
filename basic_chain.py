import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

from dotenv import load_dotenv


def get_model(repo_id="ChatGPT", **kwargs):
    if repo_id == "ChatGPT":
        chat_model = ChatOpenAI(temperature=0, **kwargs)
    else:
        raise Exception("Sorry, we only serve OpenAI models for now.")
        return None
    return chat_model


def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")

    chain = prompt | model
    return chain


def main():
    load_dotenv()

    prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")
    chain = basic_chain(prompt=prompt) | StrOutputParser()

    results = chain.invoke({"author": "William Faulkner"})
    print(results)


if __name__ == '__main__':
    main()
