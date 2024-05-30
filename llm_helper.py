from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Literal
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser


class content_insights(BaseModel):
    """Get basic details related to the content like tone, topics, keywords"""

    content_tone: Literal[
        "Formal",
        "Informal",
        "Friendly",
        "Casual",
        "Conversational",
        "Descriptive",
        "Persuasive",
        "Technical",
        "Analytical",
        "Journalese",
        "Poetic",
        "Factual",
        "Emotional",
        "Satirical",
        "Empathetic",
        "Opinionated",
        "Humorous",
        "Story-telling",
        "Narrative",
        "Expository",
        "Argumentative",
        "Objective",
        "Subjective",
    ] = Field(
        ...,
        description="The overall tone of the content",
    )
    content_topic: str = Field(
        ...,
        description="A word/phrase/sentence which captures the central topic being talked about in the content",
    )
    
    suggested_keywords: List[str] = Field(
        ...,
        description="A list of major keywords relevant to the content. Cover all major keywords"
    )
    
    search_queries : List[str] = Field(
        ...,
        description="A list of 3-4 google/search engine query for which the user provided content would be an ideal match. This search queriwa should NOT be hyper-specific neither too generic, they should contain JUST enough information which would surface this page"
    )
    
   
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are an AI assistant with an expertise in marketing content like emails, blogs, tweets and more. Follow the instructions to provide insights and details regarding the content input by the user"), 
     ("user", "Category: {category} \n Content : {content} \n\n Return the content insights for the above content")]
)   
 
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([content_insights])

parser = JsonOutputKeyToolsParser(key_name="content_insights", first_tool_only=True)

insights_chain = prompt | model | parser
