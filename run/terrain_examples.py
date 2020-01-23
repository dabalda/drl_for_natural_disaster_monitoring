from environments.envs.point_envs.surveillance import SurveillanceEnv


def main():
    log_dir = '../../drl_for_surveillance_runs/flood/terrain/'

    env = SurveillanceEnv(nr_agents=2,
                          obs_mode='normal',
                          obs_type='flood',
                          obs_radius=500,
                          world_size=1000,
                          grid_size=100,
                          range_cutpoints=30,
                          angular_cutpoints=40,
                          torus=False,
                          dynamics='aircraft',
                          shared_reward=False,
                          render_dir=log_dir + 'video/')

    env.reset()
    while True:
        env.world.reset()

if __name__ == '__main__':
    main()