import math
import os
import streamlit as st
from enum import Enum
from typing import Any, Dict, Optional, TypedDict

import instructor
import requests
import ujson
from bs4 import BeautifulSoup
from deep_translator.google import GoogleTranslator
from groq import Groq
from pydantic import AnyHttpUrl, BaseModel, Field

# Initialize Groq client
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("GROQ_API_KEY not found in secrets. Please check your configuration.")
    st.stop()


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
        list(), description="The sources of the fact check"
    )


class ConfidenceLevel(str, Enum):
    """The confidence level enum"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FactCheckResponse(BaseModel):
    """The response model for the fact check endpoint"""

    label: FactCheckLabel = Field(description="The label of the fact check")
    summary: str = Field(description="The summary of the claim")
    response: str = Field(description="The logical explanation of the fact check")
    references: list[AnyHttpUrl] = Field(description="The references of the fact check")
    confidence_score: float = Field(
        0.0, description="Confidence score from 0.0 to 1.0 based on logprobs"
    )
    confidence_level: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Qualitative confidence level (high, medium, low)",
    )


def to_english(text: str) -> str:  # type: ignore
    """translates text to english if it is not already in english."""
    text = " ".join(text.split()).rstrip(".")
    translator = GoogleTranslator(source="auto", target="en")  # type: ignore
    text = translator.translate(text)  # type: ignore
    return text


def summarize(text: str, model: str) -> str:
    """summarizes the text via Groq."""
    if len(text) <= 200:
        return text
    try:
        # client_ = instructor.from_openai(client)
        client = instructor.from_groq(groq_client)
        response = client.chat.completions.create(
            model=model,
            response_model=GPTGeneratedSummary,
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
            max_tokens=1500,
        )
        assert isinstance(response, GPTGeneratedSummary)
        return response.summary or ""
    except AssertionError:
        return text


@st.cache_data
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


def get_confidence_level(score: float) -> ConfidenceLevel:
    """Convert numerical confidence score to qualitative level"""
    if score >= 0.8:
        return ConfidenceLevel.HIGH
    elif score >= 0.5:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def calculate_confidence_from_logprobs(logprobs: Optional[Dict[str, Any]]) -> float:
    """
    Calculate confidence score based on token logprobs

    Args:
        logprobs: The logprobs object from the LLM response

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not logprobs:
        return 0.5  # Default medium confidence if no logprobs available

    # Extract token logprobs
    token_logprobs = []
    if "content" in logprobs and logprobs["content"]:
        for token_info in logprobs["content"]:
            if "logprob" in token_info:
                token_logprobs.append(token_info["logprob"])

    if not token_logprobs:
        return 0.5  # Default medium confidence if no token logprobs

    # Calculate average token probability
    avg_prob = math.exp(sum(token_logprobs) / len(token_logprobs))

    # Calculate the variance of probabilities (higher variance = lower confidence)
    probs = [math.exp(lp) for lp in token_logprobs]
    variance = sum((p - avg_prob) ** 2 for p in probs) / len(probs)

    # Calculate confidence score (higher avg_prob and lower variance means higher confidence)
    # The formula balances both factors with an emphasis on high average probability
    base_confidence = avg_prob
    variance_penalty = min(0.3, variance * 2)  # Cap the variance penalty

    confidence_score = min(1.0, max(0.0, base_confidence - variance_penalty))

    return confidence_score


def fact_check(claim: str, model: str) -> tuple[GPTFactCheckModel, float]:
    """
    fact_check checks the data against the Groq API with logprobs to calculate confidence.

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
    client = instructor.from_groq(groq_client)

    response = client.chat.completions.create(
        model=model,
        response_model=SearchQuery,
        messages=[
            {
                "role": "system",
                "content": "You are a fact-check researcher whose task is to search information to help in the fact checking. Frame an appropriate query to get the most appropriate results that will aid in the fact check",
            },
            {
                "role": "user",
                "content": claim,
            },
        ],
    )
    assert isinstance(response, SearchQuery)

    search_results = search_tool(response.query, model)

    # Send the search results back to GPT for analysis
    # Request logprobs to calculate confidence
    final_response_with_logprobs = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "I want you to act as a fact checker. You will be given a statement along with relevant search results and you are supposed to provide a fact check based on the search results. You need to classify the claim as 'correct', 'incorrect', or 'misleading' and provide the logical explanation along with the sources you used.",
            },
            {
                "role": "user",
                "content": f"Original statement: {claim}\n\nSearch results: {ujson.dumps(search_results, escape_forward_slashes=False)}",
            },
        ],
        logprobs=True,
        top_logprobs=5,
    )

    # Extract logprobs for confidence calculation
    logprobs = None
    if (
        hasattr(final_response_with_logprobs, "choices")
        and final_response_with_logprobs.choices
    ):
        logprobs = final_response_with_logprobs.choices[0].logprobs

    if logprobs is None:
        raise Exception("Lobprobd error")

    # Calculate confidence score
    confidence_score = calculate_confidence_from_logprobs(logprobs)

    # Parse the response with instructor
    response_content = final_response_with_logprobs.choices[0].message.content
    final_response = client.chat.completions.create(
        model=model,
        response_model=GPTFactCheckModel,
        messages=[
            {
                "role": "system",
                "content": "You are parsing the fact check result into a structured format.",
            },
            {"role": "user", "content": response_content or ""},
        ],
    )

    assert isinstance(final_response, GPTFactCheckModel)
    return final_response, confidence_score


def fact_check_process(
    text_data: str,
    model: str,
) -> FactCheckResponse:
    """
    fact_check_process checks the data against the Groq API.

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

    # Get confidence level from calculated score
    confidence_level = get_confidence_level(confidence_score)

    # assign to right variable
    fact_check_obj = FactCheckResponse(
        label=fact_check_resp.label,
        response=fact_check_resp.explanation,
        summary=text_data,
        references=fact_check_resp.sources,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
    )

    return fact_check_obj


@st.cache_data
def search_tool(query: str, model: str, num_results: int = 3):
    """Tool to search via Google CSE"""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    cx = os.getenv("GOOGLE_CSE_ID", "")
    base_url = "https://www.googleapis.com/customsearch/v1"
    url = f"{base_url}?key={api_key}&cx={cx}&q={query}&num={num_results}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    json = ujson.loads(resp.text)
    assert hasattr(json, "items")
    res = [get_url_content(item, model) for item in json["items"]]
    return res


def main():
    st.set_page_config(
        page_title="FactCheck AI",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("🔍 FactCheck AI")
    st.markdown("## Verify claims using AI-powered fact checking")

    # Check for required API keys
    if not all(key in st.secrets for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"]):
        st.error(
            "Missing required API keys in secrets. Please check your configuration."
        )
        st.stop()

    # User input
    claim = st.text_area(
        "Enter the claim/text to fact-check:",
        placeholder="Paste your text here...",
        height=150,
    )

    models = sorted(model.id for model in groq_client.models.list().data)
    model = st.selectbox("Choose a Model", models, index=0)

    if st.button("Verify Claim", type="primary"):
        if not claim.strip():
            st.warning("Please enter some text to verify.")
            return

        # Process the claim
        with st.spinner("Translating..."):
            translated_claim = to_english(claim)
        with st.spinner("Summarizing..."):
            summarized_claim = summarize(translated_claim, model)
        with st.spinner("Analyzing and verifying claim..."):
            result = fact_check_process(summarized_claim, model)

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
                    f"**Status:** :{color_map[result.label]}[{result.label.upper()}]"
                )

            # Confidence score display
            confidence_color_map = {
                ConfidenceLevel.HIGH: "green",
                ConfidenceLevel.MEDIUM: "blue",
                ConfidenceLevel.LOW: "orange",
            }

            confidence_percentage = int(result.confidence_score * 100)
            with col2:
                st.markdown(
                    f"**Confidence:** :{confidence_color_map[result.confidence_level]}[{result.confidence_level.upper()} ({confidence_percentage}%)]"
                )

            # Progress bar for confidence visualization
            st.progress(result.confidence_score)

            # Explanation
            st.markdown("### Explanation")
            st.write(result.response)

            st.markdown("### Confidence Analysis")
            st.write(f"""
            The model's confidence score of **{confidence_percentage}%** indicates 
            {
                "high certainty in its assessment"
                if result.confidence_level == ConfidenceLevel.HIGH
                else "moderate certainty in its assessment"
                if result.confidence_level == ConfidenceLevel.MEDIUM
                else "significant uncertainty in its assessment"
            }. 
            
            This metric is derived from analyzing the probability distributions of the generated tokens.
            """)

            # References
            if result.references:
                st.markdown("### References")
                for i, source in enumerate(result.references, 1):
                    st.markdown(f"{i}. [{source}]({source})")
            else:
                st.info("No references found for this verification")


if __name__ == "__main__":
    main()
