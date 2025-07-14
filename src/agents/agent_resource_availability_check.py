import io
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import aiofiles
from genai_session.session import GenAISession
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agents.agent_schema import AgentProperty
from src.core.config import get_settings

settings = get_settings()
session = GenAISession()

agent_property = AgentProperty(session, UUID("df0bfed8-21ea-4240-b2d6-50030a6fdfec"))


class Resource(BaseModel):
    type: Literal["bed", "stretcher"]
    number: str


class ResourceAvailability(BaseModel):
    admission_status: Literal["Accepted", "Redirected"]
    assigned_resource: list[Resource]
    assigned_doctor: str
    reason: str | None
    suggested_hospital: str


output_parser = PydanticOutputParser(pydantic_object=ResourceAvailability)
format_instructions = output_parser.get_format_instructions()


@session.bind(name="resource_availability_check", description="used to check hospital resource and travel assistance")
async def get_current_date(agent_context, emergency_sheet: Annotated[str, "patient emergency sheet case sheet"]):
    agent_context.logger.info("Inside resource_availability_check")

    async with aiofiles.open(
        Path(__file__).parent.parent / "models/data/hospital_details.json",
        "rb",
    ) as json_file:
        file_data = await json_file.read()
        hospital_list = json.load(io.BytesIO(file_data))

    async with aiofiles.open(
        Path(__file__).parent.parent / "models/data/resource_availability.json",
        "rb",
    ) as json_file:
        file_data = await json_file.read()
        resource_availability = json.load(io.BytesIO(file_data))

        human_msg = """

### ROLE
You are a hospital operations assistant responsible for managing emergency patient admissions. Based on the inputs provided below, determine if the patient can be admitted to the current hospital.
### TASK:
Based on the two JSON inputs below:

1. `resource_availability` — {resource_availability}
2. `hospital_availability` — {hospital_list}
3. `ActionRequired` - {emergency_sheet}
Your job is to:

- **Check if the current hospital can admit the patient** based on available beds or stretchers.
- If **either beds or stretchers are available**, assign the resource (bed preferred over stretcher) and one available doctor.
- If **neither beds nor stretchers are available**, suggest the **best-rated alternative hospital** that has the required department.

---
Do not make up any hospitals or departments not present in the input. Work only with the given data.
---

Output Format:
{format_instructions}
"""
        emergency_action_prompt = PromptTemplate(
            template=human_msg,
            input_variables=["resource_availability", "hospital_list", "emergency_sheet", "format_instructions"],
        )
        llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        chain = emergency_action_prompt | llm | output_parser
        chain_response = await chain.ainvoke(
            {
                "resource_availability": resource_availability,
                "hospital_list": hospital_list,
                "emergency_sheet": emergency_sheet,
                "format_instructions": format_instructions,
            }
        )
        print(chain_response)
        return chain_response.model_dump_json()


async def main():
    await session.process_events()
