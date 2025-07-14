from typing import Annotated
from uuid import UUID

from genai_session.session import GenAISession
from qdrant_client import models

from src.agents.agent_schema import AgentProperty
from src.core.config import get_settings
from src.services.qdrant.vector_db import Qdrant

settings = get_settings()
session = GenAISession()

agent_property = AgentProperty(session, UUID("e9dc5dbd-433e-4df8-ada1-cfda98440a66"))


@session.bind(name="case_history_search", description="searching patient case history from vector database")
async def get_current_date(agent_context, query: Annotated[str, "user query"]):
    agent_context.logger.info("Inside case_history_search")
    async with Qdrant(index_name=settings.QDRANT_COLLECTION) as vector_db:
        retriever = vector_db.middle_ware.as_retriever(
            search_kwargs={
                "k": settings.MAX_RELEVANT_MATCHES,
                "search_params": models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=settings.OVERSAMPLING_FACTOR,
                    )
                ),
            }
        )
        from langchain.tools.retriever import create_retriever_tool
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            """
Department: {department}
Patient History: {page_content}
TreatmentGiven: {treatment_given}
CurrentMedications: {current_medications}
Allergies: {allergies}
Vitals: {vitals}
ConsultantRecommendation: {consultant_recommendation}
caseSummary: {case_summary}
"""
        )

        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_case_history",
            "Search and return information about about patient history",
            document_prompt=prompt,
            document_separator="\n",
        )
        case_summary = retriever_tool.invoke({"query": query})
        agent_context.logger.info(case_summary)
        full_response = f"SimilarCase:\n{case_summary}"
        return full_response


async def main():
    await session.process_events()
