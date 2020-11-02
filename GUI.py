#! /usr/bin/env python
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from reference import GeneticSelector
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_gtk3agg import (
	FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np

global size, mutation_rate, crossover_rate

class MyWindow(Gtk.Window):
	def __init__(self):
		Gtk.Window.__init__(self, title="Emotion Recognition")

		# Text Boxes
		grid = Gtk.Grid()
		grid.set_column_spacing(10)
		grid.set_column_homogeneous(True)
		self.add(grid)
		# grid.attach(widget, column, row, colspan, rowspan)

		label1 = Gtk.Label()
		label1.set_text("Population Size")
		grid.attach(label1, 0, 1, 2, 1)

		label2 = Gtk.Label()
		label2.set_text("Crossover Chance")
		grid.attach(label2, 0, 2, 2, 1)

		label3 = Gtk.Label()
		label3.set_text("Mutation Chance")
		grid.attach(label3, 0, 3, 2, 1)

		# Sliders

		#ad1 = Gtk.Adjustment(initial value, min value, max value, step increment, page increment, page size)
		ad1 = Gtk.Adjustment(200, 0, 200, 10, 20, 0)
		ad2 = Gtk.Adjustment(0, 0, 100, 5, 10, 0)
		ad3 = Gtk.Adjustment(5, 0, 100, 5, 10, 0)

		# Slider for population size
		self.slide1 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad1)
		self.slide1.set_hexpand(True)
		self.slide1.connect("value-changed", self.set_populationSize)
		grid.attach(self.slide1, 2, 1, 2, 1)

		# Slider for Crossover Chance
		self.slide2 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad2)
		self.slide2.set_hexpand(True)
		self.slide2.connect("value-changed", self.set_crossoverChance)
		grid.attach(self.slide2, 2, 2, 2, 1)

		# Slider for Mutation Chance
		self.slide3 = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=ad3)
		self.slide3.set_hexpand(True)
		self.slide3.connect("value-changed", self.set_mutationChance)
		grid.attach(self.slide3, 2, 3, 2, 1)

		f = Figure(figsize=(5, 4), dpi=100)
		a = f.add_subplot(111)
		t = np.arange(0.0, 3.0, 0.01)
		s = np.sin(2*np.pi*t)
		a.plot(t, s)
		

		canvas = FigureCanvas(f)  # a Gtk.DrawingArea
		canvas.set_size_request(800, 600)
		#sw.add_with_viewport(canvas)
		grid.attach(canvas, 4, 0, 4, 4)


		# self.box = Gtk.Box(spacing=3)
		# self.add(self.box)

		button1 = Gtk.Button(label="Play")
		button1.connect("clicked", self.play_clicked)
		grid.attach(button1, 0, 0, 1, 1)

		button2 = Gtk.Button(label="Pause")
		button2.connect("clicked", self.pause_clicked)
		grid.attach(button2, 1, 0, 1, 1)

		button3 = Gtk.Button(label="Clear and Reset")
		button3.connect("clicked", self.reset_clicked)
		grid.attach(button3, 2, 0, 1, 1)

		button4 = Gtk.Button(label="DEFAULT CONFIG")
		button4.connect("clicked", self.default_clicked)
		grid.attach(button4, 3, 0, 1, 1)


	def play_clicked(self, widget):
		#GeneticSelector = GeneticSelector(estimator=LinearRegression(),
					  #n_gen=20, size, n_best=40, n_rand=40,
					  #n_children=5, mutation_rate)
		print("play")

	def pause_clicked(self, widget):
		print("pause")

	def reset_clicked(self, widget):
		size = self.slide1.set_value(0)
		crossover_rate = self.slide2.set_value(0)
		mutation_rate = self.slide3.set_value(0)
		print(crossover_rate)

	def default_clicked(self, widget):
		size = self.slide1.set_value(200)
		crossover_rate = self.slide2.set_value(5)
		mutation_rate = self.slide3.set_value(5)
		print(crossover_rate)

	def set_populationSize(self, widget):
		size = int(self.slide1.get_value())
		print(size)

	def set_crossoverChance(self, widget):
		crossover_rate = round(self.slide2.get_value() / 100 , 3)
		print(crossover_rate)

	def set_mutationChance(self, widget):
		mutation_rate = round(self.slide3.get_value() / 100 , 3)
		print(mutation_rate)


win = MyWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()

