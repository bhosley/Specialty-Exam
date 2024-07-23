"""Script for Dr. Robbins Questions
Establishing a baseline implementation of Single Agent Lunar Lander
with a DQN.

See: https://gymnasium.farama.org/environments/box2d/lunar_lander/
for more details on the environment.

How to run this script
----------------------
`python [script_name].py`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`






python dqn_exp.py \
    --num-env-runners=5 \
    --stop-reward=200 \
    --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f \
    --wandb-project=lunar_lander_test

tensorboard --logdir logs/fit
"""
# --evaluation-duration=5 \ # Default uses 10
# --checkpoint-at-end
# --num-samples=30
# result_dict['env_runners']['episode_return_mean']


import numpy as np
from gymnasium import Wrapper, spaces
from argparse import ArgumentParser

from ray import tune
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv 
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls, register_env

import multi_lander


parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for num_agents
    default_reward=600.0, 
    default_iters=500, #100, #200
    default_timesteps=1000000, #10000, #100000
)
parser.add_argument(
    "--control",
    type=str,
    choices=["Baseline", "SA", "NoPS", "FuPS"],
    default="Baseline",
    help="The controller method."
    "`Baseline`: Original Lunar Lander with single agent and lander"
    "`SA`: A single agent controls all of the landers."
    "`NoPS`: No parameter sharing between the agents."
    "`FuPS`: Full parameter sharing between the agents.",
)
parser.add_argument(
    "--SA", action="store_true",
    help="`SA`: Creates a single agent for controlling all of the agents.",
)
parser.add_argument(
    "--NoPS", action="store_true",
    help="`NoPS`: No parameter sharing between the agents.",
)
parser.add_argument(
    "--FuPS", action="store_true",
    help="`FuPS`: Full parameter sharing between the agents.",
)
parser.add_argument(
    "--num-agents", type=int, default=2,
    help="The number of agents",
)
parser.add_argument(
    "--sweep", action="store_true",
    help="Perform a parameter sweep instead of using tune",
)


class CustomWrapper(ParallelPettingZooEnv):
    """Wraps a Parallel Petting Zoo Environment for RLlib

    This wrapper is necessary to interface with rllib's workers.
    There is a problem caused when the one agent terminates before
    the other, which seems to pass inconsistent length episode 
    trajectories to the rollout worker(s).

    It also adds the necessary `__all__` key to the done dictionaries.
    """
    def step(self, action_dict) -> tuple:
        obs, rew, terminateds, truncateds, info = self.par_env.step(action_dict)

        active = [a_id for a_id,term in terminateds.items() if term==False]

        obs_s = {a_id: obs.get(a_id) for a_id in active}
        rew_s = {a_id: rew.get(a_id) for a_id in active}
        terminateds = {a_id: terminateds.get(a_id) for a_id in active}
        truncateds = {a_id: truncateds.get(a_id) for a_id in active}
        
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        return obs_s, rew_s, terminateds, truncateds, info

# Registry is necessary for functional passing later.
register_env("ma-lander", lambda _: CustomWrapper(multi_lander.Parallel_Env()))


class SingleAgentWrapper(Wrapper):
    """Wraps a parallel multi-agent environment for single-agent control.

    *It is currently limited to agents with discrete action spaces and 
    box observation spaces with identical lengths. 

    This will wrap environments with an arbitrary number of agents,
    however, the action space scales exponentially by the number of 
    agents and is likely to be the limiting factor for performance.
    """
    def __init__(self,*args):
        super().__init__(*args)
        # Concat all action spaces into a single action space
        self.action_space = spaces.Discrete(np.prod(
                            [list(self.env.action_spaces.values())[i].n 
                            for i in range(len(self.env.observation_spaces))]))
        self._act_n = list(self.env.action_spaces.values())[0].n
        # Concat all obs spaces into a single obs space
        low,high = [],[]
        for i in range(len(self.env.observation_spaces)):
            low.extend(list(self.env.observation_spaces.values())[i].low)
            high.extend(list(self.env.observation_spaces.values())[i].high)
        self.observation_space = spaces.Box(np.array(low), np.array(high))

    def observe(self) -> tuple:
        # Concat all obs values into a single obs tuple
        #return sum(self.env.observe().values(),[]) # By monoid
        return np.array(list(self.env.observe().values())).flatten()

    def observation(self) -> tuple:
        return self.observe()

    def reset(self,*, seed:int | None=None, options:dict | None=None) -> tuple:
        self.env.reset(seed=seed, options=options)
        return self.observe(), {}

    def step(self, action:int) -> tuple:
        action_list = {a: int(action/(self._act_n**i))%self._act_n
                        for i,a in enumerate(self.agents)}
        obss, rews, terminations, _, _ = self.env.step(action_list)
        obs = self.observe()
        reward = sum(rews.values())
        term = (all(terminations.values()) or self.env.game_over 
            or np.any(obs <= self.observation_space.low) 
            or np.any(obs >= self.observation_space.high))
        return obs, reward, term, False, {}

# Registry is necessary for functional passing later.
register_env("sa-lander", 
    lambda _: SingleAgentWrapper(multi_lander.Parallel_Env()))


if __name__ == "__main__":
    args = parser.parse_args()

    # Shared config settings for all experiments
    base_config = (
        get_trainable_cls("DQN")
        .get_default_config()
        .framework('torch')
    )

    # Set config for experiment if using Single Agent Control
    if args.control == 'NoPS' or args.SA:
        config = (
            base_config
            .environment(
                "sa-lander",
                env_config={"num_landers": args.num_agents}
            )
        )
    # Set config for experiment if using No-Parameter Sharing
    elif args.control == 'NoPS' or args.NoPS:
        policies = {"lander_" + str(i) for i in range(args.num_agents)}
        config = (
            base_config
            .multi_agent(
                policies=policies, # 1:1 map from AgentID to ModuleID.
                policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
            )
            .rl_module(
                rl_module_spec=MultiAgentRLModuleSpec(
                   module_specs={p:SingleAgentRLModuleSpec() for p in policies},
                ),
            )
            .environment(
                "ma-lander",
                env_config={"num_landers": args.num_agents}
            )
        )
    # Set config for experiment if using Full-Parameter Sharing
    elif args.control == 'FuPS' or args.FuPS:
        config = (
            base_config
            .multi_agent(
                policies={"p0"}, # All agents map to the same policy.
                policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
            )
            .rl_module(
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={"p0": SingleAgentRLModuleSpec()},
                ),
            )
            .environment(
                "ma-lander",
                env_config={"num_landers": args.num_agents}
            )
        )
    # Default, original lunar lander with one agent and lander
    else:
        config = (
        base_config
        .environment(
            # env="lunar-lander"
            "ma-lander",
            env_config={"num_landers": 1}
        )
    )

    # Parameter sweep settings
    if args.sweep:
        param_space = {
            "adam_epsilon": tune.loguniform(1e-4, 1e-10), # 1e-8
            "sigma0": tune.randn(0.5, 0.2), # 0.5
            "n_step": tune.choice([2**i for i in range(5)]), # 1
            # Update the target by \tau * policy + (1-\tau) * target_policy.
            "tau": tune.uniform(0.0,1.0), # 1.0,
            #epsilon = [(0, 1.0), (10000, 0.05)]    [(step,epsilon),...]
            # -> 1.0 at beginning, decreases to 0.05 over 10k steps
        }
        config = config.training(**param_space)
        # Use Param sweep, not tune
        #args.no_tune = True

    # Call experiment runner
    run_rllib_example_script_experiment(config, args)