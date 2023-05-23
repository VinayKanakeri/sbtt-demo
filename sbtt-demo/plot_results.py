import numpy as np
from matplotlib import pyplot as plt

drop_ratio = np.load('random_lorenz_time_eval_random_result_drop_ratio.npy')

# Lorenz plots
lorenz_random = np.load('random_lorenz_time_eval_random_result_mse.npy')
lorenz_uniform = np.load('uniform_lorenz_time_eval_uniform_result_mse.npy')
# lorenz_random_uniform = np.load('random_lorenz_eval_uniform_result_mse.npy')
fig, ax = plt.subplots()
ax.plot(drop_ratio, lorenz_random, linewidth=2, marker='o', label='lorenz - train:random, eval: random')
ax.plot(drop_ratio, lorenz_uniform, linewidth=2, marker='o', label='lorenz - train:uniform, eval: uniform')
# ax.plot(drop_ratio, lorenz_random_uniform, linewidth=2, marker='o', label='lorenz - train:random, eval:uniform')
ax.set_xlabel('Fraction dropped samples')
ax.set_ylabel('MSE')
ax.grid(True)
ax.set_title('Uniform and random sampling comparison for lorenz data')
ax.legend()
plt.savefig('uniform_random_lorenz.png')

# calcium2p plots
drop_ratio_calcium_uniform = np.load('uniform_CalZiGamma_time_eval_uniform_result_drop_ratio.npy')
drop_ratio_calcium_random = np.load('random_CalZiGamma_time_eval_random_result_drop_ratio.npy')
drop_ratio_calcium_staggered = np.load('staggered_CalZiGamma_time_eval_staggered_result_drop_ratio.npy')
calcium2p_random = np.load('random_CalZiGamma_time_eval_random_result_mse.npy')
calcium2p_uniform = np.load('uniform_CalZiGamma_time_eval_uniform_result_mse.npy')
calcium2p_staggered = np.load('staggered_CalZiGamma_time_eval_staggered_result_mse.npy')
fig, ax = plt.subplots()
ax.plot(drop_ratio_calcium_random, calcium2p_random, linewidth=2, marker='o', label='calcium2p - random')
ax.plot(drop_ratio_calcium_uniform, calcium2p_uniform, linewidth=2, marker='o', label='calcium2p - uniform')
ax.plot(drop_ratio_calcium_staggered, calcium2p_staggered, linewidth=2, marker='o', label='calcium2p - staggered')
ax.set_xlabel('Fraction dropped samples')
ax.set_ylabel('MSE')
ax.grid(True)
ax.set_title('Uniform and random sampling comparison for calcium2p data')
ax.legend()
plt.savefig('uniform_random_calcium2p.png')

# # calcium2p trained using ground truth loss
# calcium2p_groundTruthLoss_random = np.load('random_calcium2p_groundTruthLoss_eval_random_result_mse.npy')
# calcium2p_groundTruthLoss_uniform = np.load('uniform_calcium2p_groundTruthLoss_eval_uniform_result_mse.npy')
# fig, ax = plt.subplots()
# ax.plot(drop_ratio, calcium2p_groundTruthLoss_random, linewidth=2, marker='o', label='calcium2p_groundTruthLoss - random')
# ax.plot(drop_ratio, calcium2p_groundTruthLoss_uniform, linewidth=2, marker='o', label='calcium2p_groundTruthLoss - uniform')
# ax.set_xlabel('Fraction dropped samples')
# ax.set_ylabel('MSE')
# ax.grid(True)
# ax.set_title('Uniform and random sampling comparison for calcium2p groundTruthLoss')
# ax.legend()
# plt.savefig('./results_1/uniform_random_calcium2p_groundTruthLoss.png')

