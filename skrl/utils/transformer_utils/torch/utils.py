from typing import Any
def get_num_units(token: str | Any, num_observations, num_states, num_actions) -> str | Any:
    """Get the number of units/features a token represents.

    :param token: Token.

    :return: Number of units/features a token represents. If the token is unknown, its value will be returned as it.
    """
    num_units = {
        "ONE": 1,
        "NUM_OBSERVATIONS": num_observations,
        "NUM_STATES": num_states,
        "NUM_ACTIONS": num_actions,
        "OBSERVATIONS": num_observations,
        "STATES": num_states,
        "ACTIONS": num_actions,
    }
    token_as_str = str(token)
    if token_as_str in num_units:
        return num_units[token_as_str]
    return token