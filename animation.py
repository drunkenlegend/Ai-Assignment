import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

	pickle_in = open("plotreq.pickle","rb")
	example_dict = pickle.load(pickle_in)
	xs = []
	#print(type(example_dict))
	#print(example_dict)

	for j in range(0, len(example_dict[0])):
		xs += [j+1]

	ys = example_dict[1]

	# Draw x and y lists
	ax.clear()
	ax.plot(xs, ys) # Plot avg scores
	# Scatter plot of Best scores
	ax.scatter(xs,example_dict[0], label='Best')

	# Format plot
	plt.legend()
	plt.ylabel('Scores')
	plt.xlabel('Generation')

ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=250)
plt.show()