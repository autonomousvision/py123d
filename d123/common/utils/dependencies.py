from typing import List, Union


def check_dependencies(modules: Union[str, List[str,]], optional_name: str) -> None:
    """Checks if the given modules can be imported, otherwise raises an ImportError with a message

    :param modules: Module name or list of module names to check
    :param optional_name: Name of the optional feature
    :raises ImportError: If any of the modules cannot be imported
    """
    modules = modules if isinstance(modules, list) else [modules]
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            raise ImportError(
                f"Missing '{module}'. Install with: `pip install d123[{optional_name}]` or `pip install -e .[{optional_name}]`"
            )
