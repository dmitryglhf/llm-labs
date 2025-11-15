import asyncio
from typing import Annotated, Optional

import requests
from ddgs import DDGS
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from pydantic import AnyUrl, Field

mcp = FastMCP("lab1-tools")
md = MarkItDown()

DDGSResult = list[dict[str, str]]


def truncate_text(text_content: str, max_lines: Optional[int]) -> str:
    if max_lines is None:
        return text_content

    lines = text_content.split("\n")
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return (
            f"{truncated}\n\n... (truncated, showing {max_lines} of {len(lines)} lines)"
        )

    return text_content


@mcp.tool
async def search(
    query: Annotated[str, Field(description="Search query")],
    ctx: Context,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    page: int = 1,
    backend: str = "auto",
) -> DDGSResult | str:
    """
    DuckDuckGo text search for web pages, articles, and information.

    Args:
        query: text search query.
        region: us-en, uk-en, ru-ru, etc. Defaults to us-en.
        safesearch: on, moderate, off. Defaults to "moderate".
        timelimit: d, w, m, y. Defaults to None.
        max_results: maximum number of results. Defaults to 10.
        page: page of results. Defaults to 1.
        backend: A single or comma-delimited backends. Defaults to "auto".
    Returns:
        List of dictionaries with search results.
    """
    await ctx.info(f"Searching web: {query[:50]}...")
    results = DDGS().text(  # type: ignore
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results,
        page=page,
        backend=backend,
    )
    await ctx.info(f"Found {len(results)} results")
    return results


@mcp.tool
async def extract(
    url: Annotated[AnyUrl, Field(description="URL of the web page")],
    ctx: Context,
    max_lines: Annotated[
        Optional[int],
        Field(description="Return only first N lines of Markdown output", default=None),
    ] = None,
) -> str:
    """
    Convert web page to Markdown.

    Supports:
    - Regular web pages (HTML)
    - Web-hosted documents

    Args:
        url: URL of the web page (http:// or https://)
        max_lines: Return only first N lines of Markdown output

    Returns:
        Markdown content

    Examples:
        - extract("https://example.com/article")
        - extract("https://example.com/document.pdf")
    """

    await ctx.info("Extracting content from URL")

    def _convert():
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        )
        md = MarkItDown(requests_session=session)
        result = md.convert(str(url))
        return result.text_content

    markdown_content = await asyncio.to_thread(_convert)

    await ctx.info("Content extracted successfully")

    markdown_content = truncate_text(markdown_content, max_lines)
    return markdown_content


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
