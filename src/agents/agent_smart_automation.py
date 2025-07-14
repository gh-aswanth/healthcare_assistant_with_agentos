import io
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import aiofiles
from genai_session.session import GenAISession
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents.agent_schema import AgentProperty

session = GenAISession()

agent_property = AgentProperty(session, UUID("fa94c912-c6e4-4af4-a5e4-a3af4aaab109"))


class Sheet(TypedDict):
    history: str
    sheet: str
    criticality: str
    actions: dict
    resource_allocation: dict
    checklist: str
    verified: str
    handover_summary: str
    fallback_response: str
    appointment_details: str


class TriageResponse(BaseModel):
    criticality: Literal["HighRisk", "LowRisk"]


class VerificationResponse(BaseModel):
    verified: Literal["yes", "no"]
    fallback_response: str | None = Field(
        None, description="fallback text if user not submitted full information give proper instructions to resubmit"
    )


def triage_selection(state):
    """
    Triage patients into high-risk or low-risk zones based on defined decision rules
    from a free-text patient note. Utilizes a prompt-based approach with language
    model and structured output parsing.

    :param state: A dictionary containing input data. The "sheet" key in the dictionary
        should hold the patient's free-text note as a string.
    :type state: dict
    :return: A dictionary indicating the criticality level of the patient as
        determined by the triage process. The returned dictionary has a key
        "criticality" which maps to a string value ("HighRisk" or "LowRisk").
    :rtype: dict
    """
    output_parser = PydanticOutputParser(pydantic_object=TriageResponse)
    format_instructions = output_parser.get_format_instructions()

    triage_prompt = PromptTemplate(
        template="""
    ### ROLE
    You are an experienced emergency‑room triage nurse.

    ### TASK
    From the free‑text patient note provided, decide whether the patient belongs in the **HighRisk**, or **LowRisk** triage zone.

    ### DECISION RULES
    (Apply the most severe rule that matches; if several match, pick the highest‑priority colour.)

    **HighRisk (Immediate)**
    • Patient is **unconscious**  or  **confused**
    • ECG shows **ventricular fibrillation**  or **ST depression**
    • Blood pressure: systolic<90mmHg **OR** diastolic<60mmHg
    • Vitals explicitly described as **unstable**
    • Described **pain** or **chest pain**

    **LowRisk (Non‑urgent)**
    • None of the HighRisk conditions apply
    • ECG described as **normal**
    • Symptoms limited to **mild headache** or request for **routine check‑up**
    • Vitals explicitly described as **stable**

       **ResponseFormat**
        {format_instructions}

    ### PATIENT NOTE
    <<<
    {user_query}
    >>>
    """,
        input_variables=["user_query", "format_instructions"],
    )

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    triage_chain = triage_prompt | llm | output_parser

    response = triage_chain.invoke(
        {
            "user_query": state["sheet"],
            "format_instructions": format_instructions,
        }
    )
    return {"criticality": response.criticality}


def green_list(state):
    """
    Modifies the state dictionary to include a predefined checklist.

    This function appends a predefined checklist string, which includes "Patient Details"
    and "Presenting Complaint" sections, to the given `state` dictionary as the value of
    the key "checklist". It then returns the updated `state` dictionary.

    :param state: Dictionary to be updated with the "checklist" key.
    :type state: dict
    :return: The updated dictionary with the added "checklist" key and its associated content.
    :rtype: dict
    """
    state["checklist"] = """
Patient Details
Presenting Complaint
	"""
    return state


def red_list(state):
    """
    Updates the state dictionary by appending a checklist string under the key "checklist".
    The checklist includes information about patient details, medical history,
    current medications, and vital signs. This function modifies the input dictionary
    in place and returns the updated dictionary.

    :param state: A dictionary representing the current state of data. It should
        be mutable so that it can be updated with the checklist information.
    :return: The updated state dictionary with the appended checklist string under
        the "checklist" key.
    :rtype: dict
    """
    state["checklist"] = """
Patient Details
Medical History
Current medications
Vital Signs
"""
    return state


def verification(state):
    """
    Performs verification of a given patient's case summary based on submitted state information. The process involves
    utilizing a language model to assess the provided data and determine missing or incomplete information, requiring
    further submission for final evaluation. The function applies specific formatting and templates for accurate results.

    :param state: A dictionary containing the state of the patient's case summary, which includes:
        - checklist (str): The items required for verification.
        - criticality (str): The risk classification level of the patient.
        - sheet (str): The submitted patient note data requiring validation.
    :return: Updated state dictionary with the verification result. Specifically:
        - verified (bool): Indicates whether the submitted case sheet meets the requirements.
        - fallback_response (str): Provides additional output or guidance in case of incomplete verification.
    """
    output_parser_2 = PydanticOutputParser(pydantic_object=VerificationResponse)
    format_instructions = output_parser_2.get_format_instructions()

    verification_prompt = PromptTemplate(
        template="""
    Please review the submitted case summary.

    Patient Case Summary:
    {user_query}

    CheckList:
    {checklist_items}

    verify the checklist items are fulfilled for case sheet.

   **ResponseFormat**
    {format_instructions}

    """,
        input_variables=["checklist_items", "format_instructions", "risk_level", "user_query"],
    )

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    verification_chain = verification_prompt | llm | output_parser_2
    response = verification_chain.invoke(
        {
            "checklist_items": state["checklist"],
            "risk_level": state["criticality"],
            "format_instructions": format_instructions,
            "user_query": state["sheet"],
        }
    )
    state["verified"] = response.verified
    state["fallback_response"] = response.fallback_response
    return state


async def search_case_history(state):
    """
    Searches the case history based on the provided state and updates the state with
    the retrieved history.

    This asynchronous function communicates with an external agent to retrieve
    case history data corresponding to the `sheet` key in the provided `state`.
    The retrieved history is then stored in the `history` key of the `state`.

    :param state: A dictionary containing the current state, including a "sheet" key
        that specifies the query to retrieve history.
    :type state: dict
    :return: Updated state with history retrieved from the agent.
    :rtype: dict
    """
    agent_response = await session.send(
        message={"query": state["sheet"]}, client_id="e9dc5dbd-433e-4df8-ada1-cfda98440a66"
    )
    state["history"] = agent_response.response
    return state


async def emergency_action_list(state):
    """
    Executes an asynchronous action by sending clinical history and patient case sheet to a
    specific client and updates the state with the received emergency actions.

    :param state: Dictionary containing the patient data. Must include "history" and "sheet"
                  keys used for generating the message and performing the operation.
    :type state: dict

    :return: Updated state dictionary including the emergency actions fetched from the agent.
    :rtype: dict
    """
    agent_response = await session.send(
        message={"clinical_history": state["history"], "patient_case_sheet": state["sheet"]},
        client_id="4d2c1223-6e60-4ee5-af6a-6831fd3245e2",
    )
    state["actions"] = json.loads(agent_response.response)
    return state


async def check_resource_availability(state):
    """
    Checks the availability of a resource and updates the state with the resource allocation.

    This function sends a request containing emergency sheet information to a specified
    agent and waits for the response. The received response is then parsed and added to
    the provided state dictionary under the "resource_allocation" key.

    :param state: A dictionary containing the current state of the resource and actions
                  to be sent in the request.
    :type state: dict
    :return: Updated state with the resource allocation information.
    :rtype: dict
    """
    agent_response = await session.send(
        message={"emergency_sheet": state["actions"]}, client_id="df0bfed8-21ea-4240-b2d6-50030a6fdfec"
    )
    state["resource_allocation"] = json.loads(agent_response.response)
    return state


async def generate_summary(state):
    """
    Generates a structured clinical handover summary for a newly admitted patient based on provided resource allocation
    and action data. The summary includes sections such as Presenting Complaint, Assessment, Current Status, and Plan.

    :param state: A dictionary containing resource allocation and actions. The expected format:
                  - "resource_allocation" (str): Data about resource assignments (e.g., admission status, bed assignment).
                  - "actions" (str): Clinical actions and assessments that need to be documented.
    :return: A dictionary with the updated "handover_summary" entry containing the generated clinical handover summary.

    """
    human_msg = """
You are to generate a structured, well-formatted clinical handover summary for a newly admitted patient, using the following information:
[ResourceAllocated]
{resource_allocation}

[ActionNeedToTaken]
{actions}

Resource Allocation Data:

Admission Status
Bed Assignment
Assigned Doctor
Suggested Hospital (if any)
Clinical Action & Assessment Data:

Department and Presenting Complaint
Clinical Assessment (including vitals, risk factors, and differential diagnosis)
Interventions and Management steps already taken
Checklist of Actions Completed
Clinical Notes (including allergies and stability)
Disposition and Next Steps
Requirements for the Output:

Begin with a brief patient introduction (including status, assigned doctor, and location).
Clearly structure sections as follows (use bold headings):
Presenting Complaint
Assessment and Differential Diagnosis
Current Status and Vital Signs
Interventions and Actions Taken
Clinical Notes
Disposition and Plan
Highlight any important allergy or safety concerns.
Use bullet points where appropriate for readability.
Be concise but comprehensive, suitable for handover to another clinician.
    """

    emergency_action_prompt = PromptTemplate(template=human_msg, input_variables=["resource_allocation", "actions"])

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    chain = emergency_action_prompt | llm | StrOutputParser()
    chain_response = await chain.ainvoke(
        {
            "resource_allocation": state["resource_allocation"],
            "actions": state["actions"],
        }
    )
    state["handover_summary"] = chain_response
    print(chain_response)
    return state


async def doctor_appointment(state):
    async with aiofiles.open(
        Path(__file__).parent.parent / "models/data/resource_availability.json",
        "rb",
    ) as json_file:
        file_data = await json_file.read()
        resource_availability = json.load(io.BytesIO(file_data))
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    human_msg = """
You are provided with a patient case sheet containing relevant clinical and demographic information. Based on this, generate a structured and professional doctor appointment creation note. Your output should be formatted for use in a hospital or clinic setting.

Output Structure:
strictly follow the output format

Doctor Appointment Request

Patient Name: [Insert if available, otherwise leave blank]
Age/Gender: [Insert if available, otherwise leave blank]
Assigned Doctor: [e.g., Dr. Anil Menon]
Department: [e.g., Emergency Medicine]
Appointment Date & Time: [Insert date/time or leave blank for scheduling]
ResourceAvailable:
{resource_available}
Presenting Complaint:
{case_sheet}
    """
    appointment_action_prompt = PromptTemplate(template=human_msg, input_variables=["resource_allocation", "actions"])
    chain = appointment_action_prompt | llm | StrOutputParser()
    chain_response = await chain.ainvoke(
        {
            "resource_available": resource_availability,
            "case_sheet": state["sheet"],
        }
    )
    state["appointment_details"] = chain_response
    return state


@session.bind(
    name="smart_automation",
    description="Triage Agent to instantly assess patient symptoms and vitals, determining if emergency care is needed. Use this agent when urgent medical attention may be required.",
)
async def smart_automation(agent_context, case_sheet: Annotated[str, "Text input"]):
    """
    This function provides a state-driven automation workflow in the form of a dynamically
    compiled state graph. It aims to manage user cases by evaluating input data against
    a sequence of operational states and decision-making nodes (e.g., triage, verification,
    and resource allocation). The workflow culminates in generating an actionable summary
    or allocating appropriate resources based on the evaluated outcomes.

    :param agent_context: Contextual information which includes logging and metadata about
                          the current interaction or session.
                       Type: Typically provided by the session environment.
    :param case_sheet: The input data in the form of a text sheet, used as the basis for
                       processing through the state-driven automation.
                       Type: str (Annotated with input description as "Text input")
    :return: Returns one of the following based on workflow evaluation:
             - A summary of actions to be performed (handover summary).
             - If verification fails or resource allocation status is not accepted:
               Returns the corresponding fallback response or resource allocation state.
             Type: Typically a dict containing the workflow's outcome or response data.
    """
    agent_context.logger.info("Inside smart_automation")
    ag_builder = StateGraph(Sheet)
    ag_builder.add_node("TriageAgent", triage_selection)
    ag_builder.add_node("LowRisk", green_list)
    ag_builder.add_node("HighRisk", red_list)
    ag_builder.add_node("ResourceAvailability", check_resource_availability)
    ag_builder.add_node("Appointment", doctor_appointment)
    ag_builder.add_node("Emergency", emergency_action_list)
    ag_builder.add_node("Verification", verification)
    ag_builder.add_node("PreviousCaseHistory", search_case_history)
    ag_builder.add_node("Summary", generate_summary)

    def continue_graph(state):
        if state["verified"] == "yes":
            if state["criticality"] == "LowRisk":
                return "Appointment"
            return "PreviousCaseHistory"
        else:
            return END

    def route(state):
        if state["resource_allocation"]["admission_status"] == "Accepted":
            return "Summary"
        else:
            return END

    ag_builder.add_edge(START, "TriageAgent")
    ag_builder.add_conditional_edges("TriageAgent", lambda s: s["criticality"], ["LowRisk", "HighRisk"])
    ag_builder.add_edge("LowRisk", "Verification")
    ag_builder.add_edge("HighRisk", "Verification")
    ag_builder.add_conditional_edges("Verification", continue_graph, ["PreviousCaseHistory", "Appointment", END])
    ag_builder.add_edge("PreviousCaseHistory", "Emergency")
    ag_builder.add_edge("Emergency", "ResourceAvailability")
    ag_builder.add_conditional_edges("ResourceAvailability", route, ["Summary", END])
    ag_builder.add_edge("Summary", END)
    graph = ag_builder.compile()
    response = await graph.ainvoke({"sheet": case_sheet})

    result = ""
    if response["verified"] == "no":
        result = response["fallback_response"]
    elif response["verified"] == "yes" and response["criticality"] == "LowRisk":
        result = response["appointment_details"]
    elif response["resource_allocation"]["admission_status"] != "Accepted":
        result = response["resource_allocation"]
    else:
        result = response["handover_summary"]

    final_response = f"""
    Please use this as the final result of the automation workflow:
     {result}
    """
    return final_response


async def main():
    await session.process_events()
