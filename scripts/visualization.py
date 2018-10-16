import matplotlib.pyplot as plt
import numpy as np

def draw(to_plot, title, xlabel, ylabel, plot_type, rotation=0):
	"""
	Uses matplotlib to draw a diagram.

	Args:
	to_plot - dict - A dictionary in the form {<xtick label> : <y_value>}
	title - str - The title of the plot.
	xlabel - str - The title of the x axis.
	ylabel - str - The title of the y axis.
	plot_type - str - Either 'bar' or 'line'. Default: 'line'.
	"""
	sorted_to_plot = sorted(to_plot.items())
	plt.figure(figsize=(10,5))
	labels_vals = list(zip(*sorted_to_plot))
	if plot_type == 'bar':
		plt.bar(range(len(to_plot)), labels_vals[1])
	else:
		plt.plot(range(len(to_plot)), labels_vals[1])
	plt.xticks(range(len(to_plot)), labels_vals[0], rotation=rotation)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()


def draw_scatter_regline(x, y, title, xlabel, ylabel):
	"""
	Uses matplotlib to draw a scatter plot and the corresponding regression line.

	Args:
	x - np.array/pandas.Series - The first variable, plotted on the x axis.
	y - np.array/pandas.Series - The second variable, plotted on the y axis.
	to_plot - dict - A dictionary in the form {<xtick label> : <y_value>}
	title - str - The title of the plot.
	xlabel - str - The title of the x axis.
	ylabel - str - The title of the y axis.
	"""
	print('Correlation coeffictient: ', np.corrcoef(x, y)[0][1])
	plt.figure(figsize=(15,5))
	plt.scatter(x, y)
	a, b = np.polyfit(np.array(x), np.array(y), deg=1)
	f = lambda point: a*point + b
	line_x = np.array([min(x),max(x)])
	plt.plot(line_x, f(x), c="orange")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

