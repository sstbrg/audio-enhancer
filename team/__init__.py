"""Team management module for the audio-enhancer agent team."""

__all__ = ["TeamManager"]


def __getattr__(name: str):
    if name == "TeamManager":
        from team.manager import TeamManager
        return TeamManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
