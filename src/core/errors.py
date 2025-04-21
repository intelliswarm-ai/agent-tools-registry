class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""
    def __init__(self, message: str, tool_name: str = None, details: dict = None):
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(message)

class ToolNotFoundError(ToolExecutionError):
    """Exception raised when a requested tool is not found."""
    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool '{tool_name}' not found",
            tool_name=tool_name
        )

class ToolValidationError(ToolExecutionError):
    """Exception raised when tool parameters validation fails."""
    def __init__(self, tool_name: str, validation_errors: dict):
        super().__init__(
            message=f"Validation failed for tool '{tool_name}'",
            tool_name=tool_name,
            details={"validation_errors": validation_errors}
        )

class ToolTimeoutError(ToolExecutionError):
    """Exception raised when a tool execution times out."""
    def __init__(self, tool_name: str, timeout_seconds: int):
        super().__init__(
            message=f"Tool '{tool_name}' execution timed out after {timeout_seconds} seconds",
            tool_name=tool_name,
            details={"timeout_seconds": timeout_seconds}
        )

class ToolPermissionError(ToolExecutionError):
    """Exception raised when there are insufficient permissions to execute a tool."""
    def __init__(self, tool_name: str, required_permissions: list):
        super().__init__(
            message=f"Insufficient permissions to execute tool '{tool_name}'",
            tool_name=tool_name,
            details={"required_permissions": required_permissions}
        ) 