from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper
from prl.environment.steinberger.PokerRL import NoLimitHoldem


def make_wrapped_env(n_players, starting_stack_sizes):
    # set env_args such that new starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=starting_stack_sizes,
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env)
    env.overwrite_args(args)
    return env
