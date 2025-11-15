from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from mcp_run_python import code_sandbox
from pydantic import Field

mcp = FastMCP("python-sandbox")

RunResult = dict[str, Any]


@mcp.tool
async def execute_python(
    code: Annotated[str, Field(description="Python code to execute")],
    ctx: Context,
    dependencies: Annotated[
        Optional[list[str]],
        Field(
            description="List of Python packages to install (e.g., ['numpy', 'pandas'])",
            default=None,
        ),
    ] = None,
    globals_dict: Annotated[
        Optional[dict[str, Any]],
        Field(
            description="Global variables to make available in the code execution context",
            default=None,
        ),
    ] = None,
) -> RunResult:
    """
    Execute Python code in a secure sandboxed environment.

    Args:
        code: Python code to execute
        dependencies: Optional list of PyPI packages to install (e.g., ['numpy', 'pandas'])
        globals_dict: Optional dictionary of global variables available during execution

    Returns:
        Dictionary with execution results:
        - On success: {status: 'success', output: list[str], return_value: Any}
        - On error: {status: 'install-error' | 'run-error', output: list[str], error: str}

    Examples:
        - execute_python("print('Hello, World!')")
        - execute_python("import numpy\\nnumpy.array([1, 2, 3])", dependencies=["numpy"])
        - execute_python("x + y", globals_dict={"x": 10, "y": 20})
    """
    await ctx.info(
        f"Executing Python code in sandbox (dependencies: {dependencies or 'none'})"
    )

    async with code_sandbox(dependencies=dependencies) as sandbox:
        result = await sandbox.eval(code, globals=globals_dict)

        if result.get("status") == "success":
            await ctx.info("Code executed successfully")
        else:
            await ctx.warning(f"Code execution failed: {result.get('status')}")

        return result


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
