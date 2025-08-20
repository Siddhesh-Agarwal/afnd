import datetime
import math
from enum import Enum
from typing import List, Optional, TypedDict

import requests
import streamlit as st
import json
from openai import OpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs
from pydantic import BaseModel, Field

# Initialize OpenAI client
oai_client = None


class GPTGeneratedSummary(BaseModel):
    summary: str = Field(description="The summary of the article")


class SearchQuery(BaseModel):
    query: str


class SearchResult(TypedDict):
    title: str
    link: str
    content: str


class FactCheckLabel(str, Enum):
    """The fact check label enum"""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    MISLEADING = "misleading"


class GPTFactCheckModel(BaseModel):
    """expected result format from OpenAI for fact checking"""

    label: FactCheckLabel = Field(description="The result of the fact check")
    explanation: str = Field(description="The explanation of the fact check")
    sources: List[str] = Field(description="The sources of the fact check")


class FactCheckResponse(BaseModel):
    """The response model for the fact check endpoint"""

    label: FactCheckLabel = Field(description="The label of the fact check")
    summary: str = Field(description="The summary of the claim")
    response: str = Field(description="The logical explanation of the fact check")
    references: List[str] = Field(description="The references of the fact check")
    confidence_score: Optional[float]


def get_url_content(item: dict) -> SearchResult:
    return {
        "title": item["title"],
        "link": item["link"],
        "content": item["snippet"],
    }


def calculate_confidence_from_logprobs(logprobs: ChoiceLogprobs) -> float:
    """Calculate confidence score based on token logprobs

    Args:
        logprobs: The logprobs object from the LLM response

    Returns:
        float: Confidence score between 0.0 and 1.0

    """
    if logprobs.content is None:
        st.error("Logprobs doesn't work with this model")
        return 0.0

    try:
        min_prob = min(math.exp(logprob.logprob) for logprob in logprobs.content)
        return min_prob
    except (ValueError, TypeError) as e:
        st.error(f"Error calculating confidence: {str(e)}")
        return 0.0


def fact_check(claim: str, model: str) -> tuple[GPTFactCheckModel, Optional[float]]:
    """fact_check checks the data against the OpenAI API with logprobs to calculate confidence.

    Parameters
    ----------
    claim : str
        The claim to be checked.
    model : str
        The model to use for fact checking.

    Returns
    -------
    tuple[GPTFactCheckModel, float]
        The fact check result and confidence score.

    """
    global oai_client

    if not oai_client:
        st.error("OpenAI client not initialized. Please provide an API key.")
        st.stop()

    response = oai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a fact-check researcher whose task is to search information to help in the fact checking. Frame an appropriate query to get the most appropriate results that will aid in the fact check. Today's date is {datetime.date.today().strftime('%d/%m/%Y')}.",
            },
            {
                "role": "user",
                "content": claim,
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "search":
                args = json.loads(tool_call.function.arguments)
                query = args["query"]
                st.write(f"Searching for: `{query}`")
                search_results = search_tool(query)
                break
        else:
            st.error("AI Agent failed")
            st.stop()
    else:
        st.error("AI Agent failed")
        st.stop()

    # Send the search results back to GPT for analysis
    resp = oai_client.beta.chat.completions.parse(
        model=model,
        response_format=GPTFactCheckModel,
        messages=[
            {
                "role": "system",
                "content": "You are a professional fact checker. You will be given a statement along with relevant search results and you are supposed to provide a fact check based on the search results. You need to classify the claim as 'correct', 'incorrect', or 'misleading' and provide the logical explanation along with the sources you used.",
            },
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nSearch results: {json.dumps(search_results)}",
            },
        ],
    )

    # Try to parse the response as JSON, with error handling
    res = GPTFactCheckModel.model_validate_json(resp.choices[0].message.content or "{}")
    return res, None


def fact_check_process(
    text_data: str,
    model: str,
) -> FactCheckResponse:
    """fact_check_process checks the data against the OpenAI API.

    Parameters
    ----------
    text_data : str
        The data to be checked.
    model : str
        The model to use.

    Returns
    -------
    FactCheckResponse
        The result of the fact check with confidence score based on logprobs.

    """
    fact_check_resp, confidence_score = fact_check(text_data, model)

    # assign to right variable
    fact_check_obj = FactCheckResponse(
        label=fact_check_resp.label,
        response=fact_check_resp.explanation,
        summary=text_data,
        references=fact_check_resp.sources,
        confidence_score=confidence_score,
    )

    return fact_check_obj


def search_tool(query: str, num_results: int = 3) -> list[SearchResult]:
    """Tool to search via Google CSE"""
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    cx = st.secrets.get("GOOGLE_CSE_ID", "")
    base_url = "https://www.googleapis.com/customsearch/v1"
    url = f"{base_url}?key={api_key}&cx={cx}&q={query}&num={num_results}"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        json_data = resp.json()

        # Check if 'items' exists in the parsed JSON
        if "items" not in json_data:
            st.warning("No search results found")
            return []

        # res = [get_url_content(item) for item in json_data["items"]]
        res = list(map(get_url_content, json_data["items"]))
        return res
    except requests.exceptions.RequestException as e:
        st.error(f"Search request error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def main():
    global oai_client

    st.set_page_config(
        page_title="FactCheck AI",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("🔍 FactCheck AI")
    st.markdown("## Verify claims using AI-powered fact checking")

    # API Key Input
    oai_api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
    )

    oai_base_url = st.text_input(
        "Enter your OpenAI Base URL:",
        value="https://api.openai.com/v1",
    )

    # Check for required Google API keys
    if not all(key in st.secrets for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"]):
        st.error(
            "Missing required Google API keys in secrets. Please check your configuration.",
        )
        st.stop()

    # Initialize OpenAI client
    if not oai_api_key:
        st.warning("Please enter your API key to continue")
        st.stop()
    elif not oai_base_url:
        st.warning("Please enter your base URL to continue")
        st.stop()
    try:
        oai_client = OpenAI(base_url=oai_base_url, api_key=oai_api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e!s}")
        st.stop()

    # Get models and select a sensible default (gpt-4 if available)
    models = sorted(model.id for model in oai_client.models.list().data)
    model = st.selectbox("Choose a Model", models, index=0)

    # User input
    claim = st.text_area(
        "Enter the claim/text to fact-check:",
        placeholder="Paste your text here...",
        height=150,
    )

    if st.button("Verify Claim", type="primary"):
        if not claim.strip():
            st.warning("Please enter some text to verify.")
            st.stop()

        # Process the claim
        with st.spinner("Analyzing and verifying claim..."):
            result = fact_check_process(claim, model)

            # Create columns for results
            col1, col2 = st.columns(2)

            # Display results
            st.subheader("Verification Result")

            # Label display with color coding
            color_map = {
                FactCheckLabel.CORRECT: "green",
                FactCheckLabel.INCORRECT: "red",
                FactCheckLabel.MISLEADING: "orange",
            }
            with col1:
                st.markdown(
                    f"**Status:** :{color_map[result.label]}[{result.label.upper()}]",
                )

            # Confidence score display
            with col2:
                if result.confidence_score is None:
                    st.warning("Confidence score not available")
                else:
                    confidence_percentage = int(result.confidence_score * 100)
                    st.metric(
                        label="Confidence",
                        value=f"{confidence_percentage}%",
                    )

            # Explanation
            st.markdown("### Explanation")
            st.write(result.response)

            # References
            st.markdown("### References")
            if result.references:
                for i, source in enumerate(result.references, 1):
                    st.markdown(f"{i}. [{source}]({source})")
            else:
                st.warning("No references found for this verification")


if __name__ == "__main__":
    main()
