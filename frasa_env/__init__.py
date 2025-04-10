from gymnasium.envs.registration import register

register(
    id="frasa-standup-v0",
    entry_point="frasa_env.env:FRASAEnv",
)
