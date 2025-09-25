import matplotlib.pyplot as plt

# example
epsilons = [1.5, 3.0, 6.0] # privacy budgets
accuracies = [72.4, 78.1, 82.9] # test accuracies (%)
leakage = [0.05, 0.12, 0.32] # canary confidence scores

fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("Privacy Budget (ε)")
ax1.set_ylabel("Accuracy (%)", color=color)
ax1.plot(epsilons, accuracies, marker="o", color=color, label="Accuracy")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # second axis for leakage
color = "tab:red"
ax2.set_ylabel("Leakage Score", color=color)
ax2.plot(epsilons, leakage, marker="x", color=color, label="Leakage")
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.title("Privacy-Utility Tradeoff in PrivacyLens")
plt.show()
