import matplotlib.pyplot as plt

shots = [10, 25, 50, 75, 100]
frozen = [75.07, 77.89, 88.86, 88.56, 90.78]
full_ft = [88.86, 98.21, 99.45, 98.36, 99.68]
scratch = [93.88, 94.76, 92.81, 97.55, 91.13]

plt.figure(figsize=(8,6), dpi=300, constrained_layout=True)

plt.plot(shots, frozen, 'o-', label='Frozen Backbone', linewidth=2)
plt.plot(shots, full_ft, 's-', label='Full Fine-Tune (ours)', linewidth=2)
plt.plot(shots, scratch, '^-', label='Scratch', linewidth=2)

plt.xlabel('Shots per Class (Total Labels = 2×Shots)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Few-Shot Scaling on Mendeley Dataset', fontsize=16)

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('few_shot_scaling_curve.png', dpi=300)
plt.show()