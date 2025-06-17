from gymnasium.envs.registration import register

register(
    id="unified-humanoid-get-up-env-standup-v0",
    entry_point="unified_humanoid_get_up_env.env:UnifiedHumanoidGetUpEnv",
)
