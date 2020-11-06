import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from matplotlib.backends.backend_gtk3agg import (
	FigureCanvasGTK3Agg as FigureCanvas)


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
listdump = [[], []]

pickle_out = open("plotreq.pickle","wb")
pickle.dump(listdump, pickle_out)
pickle_out.close()
	
# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

		pickle_in = open("plotreq.pickle","rb")
		example_dict = pickle.load(pickle_in)
		xs = []

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

ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=500)
plt.show()
