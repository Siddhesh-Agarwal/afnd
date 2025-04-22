import asyncio
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


def summarize(text: str, model: str) -> str:
    """summarizes the text via Groq."""
    if len(text) <= 200:
        return text
    try:
        global oai_client

        if not oai_client:
            st.error("OpenAI client not initialized. Please provide an API key.")
            st.stop()

        response = oai_client.beta.chat.completions.parse(
            model=model,
            response_format=GPTGeneratedSummary,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise summary in the language of the article. ",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text in a concise way:\n{text}",
                },
            ],
            max_tokens=1000,
        )
        assert isinstance(response, GPTGeneratedSummary)
        return response.summary or ""
    except AssertionError:
        return ""


def get_content(url: str, model: str) -> str | None:
    """returns the content of given url"""
    try:
        with requests.get(url, timeout=15) as res:
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            return summarize(soup.get_text(), model)
    except requests.exceptions.RequestException:
        return None


def get_url_content(item: dict, model: str) -> SearchResult:
    content = get_content(str(item.get("link", "")), model)
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
        return -1
    min_prob = min(math.exp(logprob.logprob) for logprob in logprobs.content)
    return min_prob


async def fact_check(claim: str, model: str) -> tuple[GPTFactCheckModel, float]:
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

    response = oai_client.beta.chat.completions.parse(
        model=model,
        response_format=SearchQuery,
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
    )
    assert isinstance(response, SearchQuery)

    search_results = await search_tool(response.query, model)

    # Send the search results back to GPT for analysis
    # Request logprobs to calculate confidence
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

    res = GPTFactCheckModel.model_validate_json(resp.choices[0].message.content or "")
    return res, confidence_score


async def fact_check_process(
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
    fact_check_resp, confidence_score = await fact_check(text_data, model)

    # assign to right variable
    fact_check_obj = FactCheckResponse(
        label=fact_check_resp.label,
        response=fact_check_resp.explanation,
        summary=text_data,
        references=fact_check_resp.sources,
        confidence_score=confidence_score,
    )

    return fact_check_obj


async def search_tool(
    query: str, model: str, num_results: int = 3
) -> list[SearchResult]:
    """Tool to search via Google CSE"""
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    cx = st.secrets.get("GOOGLE_CSE_ID", "")
    base_url = "https://www.googleapis.com/customsearch/v1"
    url = f"{base_url}?key={api_key}&cx={cx}&q={query}&num={num_results}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    json = ujson.loads(resp.text)
    assert hasattr(resp, "items")
    res = [get_url_content(item, model) for item in json["items"]]
    return res


async def main_async():
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
    if not oai_api_key:
        st.warning("Please enter your OpenAI API key to continue")
        st.stop()

    # Initialize OpenAI client
    if oai_api_key:
        try:
            oai_client = OpenAI(api_key=oai_api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e!s}")
            st.stop()

    # Check for required Google API keys
    if not all(key in st.secrets for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"]):
        st.error(
            "Missing required Google API keys in secrets. Please check your configuration.",
        )
        st.stop()

    # User input
    claim = st.text_area(
        "Enter the claim/text to fact-check:",
        placeholder="Paste your text here...",
        height=150,
    )

    if oai_client is None:
        st.stop()

    models = sorted(model.id for model in oai_client.models.list().data)
    model = st.selectbox("Choose a Model", models, index=0)

    if st.button("Verify Claim", type="primary"):
        if not claim.strip():
            st.warning("Please enter some text to verify.")
            st.stop()

        if not oai_client:
            st.error("OpenAI client not initialized. Please provide a valid API key.")
            st.stop()

        # Process the claim
        with st.spinner("Analyzing and verifying claim..."):
            result = await fact_check_process(claim, model)

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
                    value=confidence_percentage,
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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
