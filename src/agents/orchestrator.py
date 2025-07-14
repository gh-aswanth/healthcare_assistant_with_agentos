"""
Agent orchestrator
"""

import asyncio
import contextlib
import logging
from typing import Dict, List, Optional

from src.core.config import get_settings
from src.services.agentos.http import http_repo

from .agent_case_history import agent_property as agent_c_property
from .agent_emergency_checklist import agent_property as agent_d_property
from .agent_resource_availability_check import agent_property as agent_e_property
from .agent_schema import Agent, AgentProperty
from .agent_smart_automation import agent_property as agent_f_property

settings = get_settings()
agent_properties = [
    agent_c_property,
    agent_d_property,
    agent_e_property,
    agent_f_property,
]

logger = logging.getLogger(__name__)


class AgentStreamLine:
    def __init__(self, agent_pool: List[Agent], username: str, password: str):
        self.agent_pool = agent_pool
        self.access_token: Optional[str] = None
        self.username = username
        self.password = password

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return authorization headers with the current access token."""
        return {"Authorization": f"Bearer {self.access_token}"}

    async def register_user(self) -> None:
        """Register a new user, suppressing any exceptions."""
        with contextlib.suppress(Exception):
            await http_repo.register_user(self.username, self.password)

    async def login(self) -> None:
        """Login and store the access token."""
        response = await http_repo.login_user(self.username, self.password)
        self.access_token = response.access_token

    async def _process_existing_agent(self, agent: Agent) -> None:
        """Handle an agent that already exists in the system."""
        try:
            response = await http_repo.lookup_agent(agent_id=agent.agent_id, headers=self._get_auth_headers())
            agent.source.jwt_token = response.agent_jwt
            logger.info(f"Agent registration skipped for {agent.agent_id}")
        except Exception as e:
            logger.error(f"Error looking up agent {agent.agent_id}: {e}")
            await self._create_new_agent(agent)

    async def _create_new_agent(self, agent: Agent) -> None:
        """Create a new agent in the system."""
        try:
            response = await http_repo.register_agent(
                agent_id=agent.agent_id,
                name=agent.source.agent.name,
                description=agent.source.agent.description,
                headers=self._get_auth_headers(),
            )
            agent.source.jwt_token = response.json()["jwt"]
            logger.info(f"Agent {agent.agent_id} registered successfully")
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")

    async def register_agent(self) -> None:
        """Register all agents in the agent pool and process their events."""
        # Ensure we have authentication
        await self.register_user()
        await self.login()

        # Register each agent
        for agent in self.agent_pool:
            await self._process_existing_agent(agent)

        # Process events for all agents concurrently
        await asyncio.gather(*[agent.source.process_events() for agent in self.agent_pool])


def _create_agent_from_property(agent_property: AgentProperty) -> Agent:
    """Create an Agent instance from an AgentProperty."""
    return Agent(agent_id=str(agent_property.agent_id), source=agent_property.session)


def _get_agent_pool() -> List[Agent]:
    """Create and return the pool of agents."""
    return [_create_agent_from_property(prop) for prop in agent_properties]


async def agent_orchestrator():
    """Orchestrate the registration and operation of all agents."""
    agent_pool = _get_agent_pool()

    await AgentStreamLine(
        agent_pool=agent_pool,
        username=settings.USERNAME,
        password=settings.PASSWORD,
    ).register_agent()


if __name__ == "__main__":
    agent_orchestrator()
