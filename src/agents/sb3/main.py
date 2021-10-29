"""Example of evaluating sb3 checkpoints."""
import zipfile
import os.path as osp
import numpy as np
import torch

from stable_baselines3 import PPO

from semantic_locomotion.intermediate_envs import gait_change_env
from semantic_locomotion.intermediate_envs.configs import pronk_deluxe
from semantic_locomotion.agents.sb3 import env_wrappers

if __name__ == "__main__":
  model_path = 'semantic_locomotion/agents/sb3/best_logs/eval/model'
  with zipfile.ZipFile(osp.join(model_path, 'best_model.zip')) as zip_ref:
    zip_ref.extractall(model_path)

  env = gait_change_env.GaitChangeEnv(config=pronk_deluxe.get_config(),
                                      use_real_robot=False,
                                      show_gui=True)
  env = env_wrappers.LimitDuration(env, 400)
  env = env_wrappers.RangeNormalize(env)

  policy_kwargs = dict(net_arch=[dict(pi=[256], vf=[256, 256])],
                       activation_fn=torch.nn.Tanh,
                       log_std_init=np.log(0.5))
  model = PPO("MlpPolicy",
              env,
              n_steps=int(4096 / 16),
              max_grad_norm=1.,
              learning_rate=1e-3,
              policy_kwargs=policy_kwargs,
              verbose=1,
              device='cpu')
  params = torch.load(osp.join(model_path, 'policy.pth'))
  model.policy.load_state_dict(params)

  obs = env.reset()
  sum_reward = 0
  while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    sum_reward += reward
    if done:
      break
  print("Total reward is: {}".format(sum_reward))
