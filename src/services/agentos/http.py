import traceback
from typing import Optional, Type, TypeVar, Union

import httpx
import typer
from pydantic import BaseModel, ValidationError

from src.core.config import get_settings

from .exceptions import (
    APIError,
    MismatchingExpectedStatusCodeError,
)
from .log import render_error, render_success
from .schemas import AccessToken, AgentSchema, RegisterResponse

settings = get_settings()


T = TypeVar("T", bound=BaseModel)


def http_client(headers: dict = None, timeout: Optional[int] = 60) -> httpx.AsyncClient:
    timeout = httpx.Timeout(timeout=timeout)
    client = httpx.AsyncClient(base_url=settings.CLI_BACKEND_ORIGIN_URL, headers=headers, timeout=timeout)
    return client


class TokenPayload(BaseModel):
    exp: Optional[int] = None
    sub: Optional[str] = None


class HTTPRepository:
    def __init__(self):
        self.client = http_client()

    async def _request(
        self,
        method: str,
        url: str,
        expected_status_code: int,
        parse_as: Optional[Type[T]] = None,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[httpx.Timeout] = httpx.Timeout(60),
    ) -> Optional[Union[T, httpx.Response]]:
        """ """
        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == expected_status_code:
                if not parse_as:
                    return response

                try:
                    # check against responses without content like 204 on delete
                    if response.content:
                        # returning pydantic model with the response
                        return parse_as(**response.json())

                    elif issubclass(parse_as, BaseModel):
                        # TODO: moar validation
                        return None

                    else:
                        # if parse_as is not a pydantic model
                        return None

                except ValidationError as e:
                    raise APIError(  # noqa: B904
                        f"Failed to cast HTTP response from {url}, method={method} to pydantic model: {parse_as.__class__.__name__}\n{e.json()}",  # noqa: E501
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                except Exception:
                    render_error(f"Unknown error occured: {traceback.format_exc()}")
                    typer.Exit(1)
            else:
                err_details = response.text
                raise MismatchingExpectedStatusCodeError(
                    f"API request failed for {url}, method={method}",
                    status_code=response.status_code,
                    response_body=err_details,
                )
        except httpx.RequestError as e:
            raise APIError(f"HTTP request failed: {e}")  # noqa: B904

    async def login_user(self, username: str, password: str) -> Optional[AccessToken]:
        token_data = await self._request(
            method="POST",
            url="/api/login/access-token",
            data={"username": username, "password": password},
            parse_as=AccessToken,
            expected_status_code=200,
        )
        if not token_data.access_token:
            raise APIError(
                "HTTP request was successful, but unexpectedly login response was malformed",
                status_code=200,
                response_body=token_data.model_dump_json(indent=4),
            )
        render_success("User logged in successfully!")
        return token_data

    async def register_user(self, username: str, password: str) -> None:
        register_data = await self._request(
            method="POST",
            url="/api/register",
            json={"username": username, "password": password},
            parse_as=RegisterResponse,
            expected_status_code=200,
        )
        if not register_data:
            raise APIError(
                "HTTP request was successful, but unexpectedly register response was malformed",
            )

        token_data = await self.login_user(username, password)
        if token_data:
            render_success(f"User '{register_data.username}' created and logged in successfully.")
            return

        raise APIError("HTTP request was successful, but unexpectedly login response was malformed")

    async def register_agent(
        self, agent_id: str, name: str, description: str, headers: Optional[dict] = None
    ) -> httpx.Response:
        if headers is None:
            headers = {}
        response = await self._request(
            method="POST",
            url="/api/agents/register",
            json={
                "id": agent_id,
                "name": name,
                "description": description,
                "input_parameters": {},
            },
            expected_status_code=200,
            headers=headers,
        )
        return response

    async def lookup_agent(self, agent_id: str, headers: Optional[dict] = None) -> Optional[AgentSchema]:
        if headers is None:
            headers = {}
        response = await self._request(
            url=f"/api/agents/{agent_id}",
            method="GET",
            expected_status_code=200,
            parse_as=AgentSchema,
            headers=headers,
        )
        if not response:
            raise APIError("HTTP request was successful, but unexpectedly agent response was malformed")
        return response


http_repo = HTTPRepository()
