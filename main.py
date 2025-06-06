import datetime
import math
from enum import Enum
from typing import TypedDict

from bs4 import BeautifulSoup
import requests
import streamlit as st
import ujson
from openai import OpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs
from pydantic import AnyHttpUrl, BaseModel, Field

# Initialize OpenAI client
oai_client = None


class GPTGeneratedSummary(BaseModel):
    summary: str | None = Field(None, description="The summary of the article")


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
    explanation: str = Field("", description="The explanation of the fact check")
    sources: list[AnyHttpUrl] = Field(
        list(),
        description="The sources of the fact check",
    )


class FactCheckResponse(BaseModel):
    """The response model for the fact check endpoint"""

    label: FactCheckLabel = Field(description="The label of the fact check")
    summary: str = Field(description="The summary of the claim")
    response: str = Field(description="The logical explanation of the fact check")
    references: list[AnyHttpUrl] = Field(description="The references of the fact check")
    confidence_score: float = Field(
        0.0,
        description="Confidence score from 0.0 to 1.0 based on logprobs",
    )


def get_content(url: str) -> str | None:
    """returns the content of given url"""
    try:
        with requests.get(url, timeout=15) as res:
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text()
    except requests.exceptions.RequestException:
        return None


def get_url_content(item: dict) -> SearchResult:
    content = get_content(str(item.get("link", "")))
    return {
        "title": item["title"],
        "link": item["link"],
        "content": content or item["snippet"],
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


def fact_check(claim: str, model: str) -> tuple[GPTFactCheckModel, float]:
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
    assert isinstance(response, SearchQuery)

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "search":
                args = ujson.loads(tool_call.function.arguments)
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
    # Request logprobs to calculate confidence
    try:
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
                    "content": f"Claim: {claim}\n\nSearch results: {ujson.dumps(search_results, escape_forward_slashes=False, indent=2)}",
                },
            ],
            logprobs=True,
        )

        logprobs = resp.choices[0].logprobs

        if logprobs is None:
            raise Exception("Logprobs error")

        # Calculate confidence score
        confidence_score = calculate_confidence_from_logprobs(logprobs)

        # Try to parse the response as JSON, with error handling
        try:
            res = GPTFactCheckModel.model_validate_json(
                resp.choices[0].message.content or "{}"
            )
            return res, confidence_score
        except Exception as e:
            st.error(f"Error parsing model response: {str(e)}")
            # Return a default model with an error explanation
            return GPTFactCheckModel(
                label=FactCheckLabel.MISLEADING,
                explanation=f"Error processing response: {str(e)}. The model response could not be parsed.",
                sources=[],
            ), 0.0

    except Exception as e:
        st.error(f"Fact checking error: {str(e)}")
        return GPTFactCheckModel(
            label=FactCheckLabel.MISLEADING,
            explanation=f"Error during fact checking: {str(e)}",
            sources=[],
        ), 0.0


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
        json_data = ujson.loads(resp.text)

        # Check if 'items' exists in the parsed JSON
        if "items" not in json_data:
            st.warning("No search results found")
            return []

        res = [get_url_content(item) for item in json_data["items"]]
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
        help="You can get your API key from https://console.openai.com/keys",
    )

    # Check for required Google API keys
    if not all(key in st.secrets for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"]):
        st.error(
            "Missing required Google API keys in secrets. Please check your configuration.",
        )
        st.stop()

    # Initialize OpenAI client
    if oai_api_key:
        try:
            oai_client = OpenAI(api_key=oai_api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e!s}")
            st.stop()
    else:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()

    # User input
    claim = st.text_area(
        "Enter the claim/text to fact-check:",
        placeholder="Paste your text here...",
        height=150,
    )

    # Get models and select a sensible default (gpt-4 if available)
    models = sorted(model.id for model in oai_client.models.list().data)
    model = st.selectbox("Choose a Model", models, index=0)

    if st.button("Verify Claim", type="primary"):
        if not claim.strip():
            st.warning("Please enter some text to verify.")
            st.stop()

        # Process the claim
        with st.spinner("Analyzing and verifying claim..."):
            try:
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
                confidence_percentage = int(result.confidence_score * 100)
                with col2:
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
            except Exception as e:
                st.error(f"An error occurred during verification: {str(e)}")


if __name__ == "__main__":
    main()
