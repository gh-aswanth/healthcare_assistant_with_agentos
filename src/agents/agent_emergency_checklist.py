from typing import Annotated
from uuid import UUID

from genai_session.session import GenAISession
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agents.agent_schema import AgentProperty
from src.core.config import get_settings

settings = get_settings()
session = GenAISession()


class EmergencyChecklist(BaseModel):
    department: str
    presenting_complaint: str
    assessment: str
    interventions_management: list[str]
    checklist_actions_taken: list[str]
    clinical_notes: str
    disposition_next_steps: str


# 2. Output parser and format instructions
output_parser = PydanticOutputParser(pydantic_object=EmergencyChecklist)
format_instructions = output_parser.get_format_instructions()

agent_property = AgentProperty(session, UUID("4d2c1223-6e60-4ee5-af6a-6831fd3245e2"))


@session.bind(
    name="emergency_checklist",
    description="used to generate emergency checklist from patient case sheet and similar cases",
)
async def get_current_date(
    agent_context,
    clinical_history: Annotated[str, "similar cases"],
    patient_case_sheet: Annotated[str, "patient case sheet"],
):
    agent_context.logger.info("Inside emergency_checklist")

    human_msg = """

Instructions:

Reference historical treatment details only to inform clinical context and best practices.
Summarize the patient's presenting complaint, assessment, interventions, and disposition in a clear, structured format.
Provide a checklist of key emergency actions taken.
Highlight any notable observations or changes in patient status.
Make recommendations for immediate next steps or disposition (admit, discharge, observation, etc.).
Exclude all personal information and focus solely on clinical decision-making.

Clinical History:
{clinical_history}

Patient CaseSheet:
{patient_case_sheet}

Structured Response Format:

Emergency Treatment Note

Department:
[Insert relevant department, e.g., Emergency Medicine]

Presenting Complaint:
[Brief summary of chief complaint based on clinical information provided]

Assessment:
[Summarize initial assessment steps and key findings, referencing historical best practices]

Interventions/Management:
[List all emergency interventions and management provided, referencing similar historical cases where relevant]

Checklist of Actions Taken:
[List all key emergency actions taken during the current assessment]

Clinical Notes:
[Document notable observations, changes in patient status, or critical considerations based on historical reference]

Disposition/Next Steps:
[Recommend admit/discharge/further observation/other actions; justify decision per historical reference]

Additional Guidance:

Only use the provided clinical information and relevant historical case data to guide your response.
Exclude all personal identifiers from your note.
Focus on immediate emergency care and support for the treating physician.

Output Format:
{format_instructions}

Return the entire response as a single JSON object with clearly defined keys corresponding to each section in the structured response above. Use arrays where multiple items are appropriate (e.g., checklist actions). Return the entire response as a single, well-formatted JSON object with clearly defined keys corresponding to each section in the structured response above. Use arrays of strings where multiple items are appropriate (e.g., checklist actions, interventions). Ensure all values are either strings or arrays of strings, and the JSON is properly structured.
"""
    emergency_action_prompt = PromptTemplate(
        template=human_msg, input_variables=["clinical_history", "patient_case_sheet", "format_instructions"]
    )
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    chain = emergency_action_prompt | llm | output_parser
    chain_response = await chain.ainvoke(
        {
            "clinical_history": clinical_history,
            "patient_case_sheet": patient_case_sheet,
            "format_instructions": format_instructions,
        }
    )
    print(chain_response)
    return chain_response.model_dump_json()


async def main():
    await session.process_events()
