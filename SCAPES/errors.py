class ConfigError(ValueError):
    """A gin file is missing a required line, has an invalid value or uses a rejected key."""
