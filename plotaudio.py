from gi.repository import Gtk, GObject, Gdk
import traceback
from pitch import PitchFinder, FormantFinder
import numpy as np
import copy
import cairo
import threading
import random
import time
import math
from multiprocessing import Process, Queue, Lock
import pyaudio
import struct
import numpy
import plotqueue
import lpc
import math
import pyautogui

CHUNKS_IN_VIEW = 200
FRAME_RATE_MS = 60.0
SUMMARY_BUFFER_MS = 50

class State(object):
    pass

class PlotWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Live Plot")
        self.connect("delete-event", self.on_delete)
        self.done = False

        hb = Gtk.HeaderBar()
        hb.props.title = "Live Plot"
        self.set_titlebar(hb)
        self.state = State()
        self.state.summary = 0
        self.state.history = np.zeros(CHUNKS_IN_VIEW)
        self.delta = .1

        self.box = Gtk.Box(spacing=6)
        self.add(self.box)

        self.plot_area = Gtk.DrawingArea()
        self.plot_area.connect("draw", self.draw_plot)

        self.plot_area.set_size_request(100,100)
        self.box.pack_start(self.plot_area, True, True, 0)
        self.color1 = (random.random(), random.random(), random.random())
        self.color2 = (random.random(), random.random(), random.random())

    def on_delete(self, *args):
        #print "Got delete", args
        self.done = True
        return False

    def draw_plot(self, widget, cr):
        t0 = time.time()

        oscope_width_px = 2000
        oscope_duration_ms = 4000 # ms
        oscope_height_px = 500

        slice_width_px = oscope_width_px / CHUNKS_IN_VIEW

        white = (255, 255, 255)
        cr.rectangle(0, 100, oscope_width_px, oscope_height_px)
        cr.set_source_rgb(*white)
        cr.fill()

        for i,c in enumerate(self.state.history[0:CHUNKS_IN_VIEW]):
            dot_y_pos =  100 + oscope_height_px - c  / 3
            cr.new_sub_path()
            cr.arc(i * slice_width_px, dot_y_pos, slice_width_px/2, 0, 2*math.pi)
            cr.close_path()
        cr.set_source_rgb(*self.color2)
        cr.fill()

        cr.set_source_rgb(*self.color1)
        cr.rectangle(0, 0, self.state.summary * 3, 100)
        cr.fill()
        #cr.set_source_rgb(*self.color2)
        #cr.rectangle(0, 110, (1+math.sin(self.n))*250, 100)
        #cr.fill()
        #t1 = time.time()
        #print "Time to draw", (t1 - t0)

    def on_button1_clicked(self, widget):
        print("Hello")

    def on_button2_clicked(self, widget):
        print("Goodbye")

def ring_in(a, b):
    return np.concatenate((np.roll(a, -1*len(b))[:len(a)-len(b)], b))

def timed_redraw(win, plot, q):
    v = 0
    vals = np.zeros(int(SUMMARY_BUFFER_MS * 1.0 / 1000 * RATE))
    #print "vals len", len(vals)
    CHUNK_NUM  = 0
    while True:

        time.sleep(1 / FRAME_RATE_MS)
        if win.done: break

        while not q.empty():
            val =  q.get(False)
            CHUNK_NUM += 1
            #rms = np.sqrt(val.dot(val)/val.size) * 100
            #pitch = PitchFinder.analyze(val, RATE)
            tracked_val = FormantFinder.analyze(val, RATE)
            win.state.history[CHUNK_NUM % CHUNKS_IN_VIEW] = tracked_val #rms
            vals = ring_in(vals, val)

        #rms = np.sqrt(vals.dot(vals)/vals.size) * 100
        #tracked_val = PitchFinder.analyze(vals, RATE)
        tracked_val = FormantFinder.analyze(vals, RATE)
        if tracked_val > 0:
            win.state.summary = tracked_val #rms * 50

        Gdk.threads_enter()
        plot.queue_draw()
        Gdk.threads_leave()

        MIN_TRACKED = 220
        MAX_TRACKED = 440 #* math.pow(2, 1.0/3)

        BOTTOM_PIXEL = 1400
        TOP_PIXEL = 130
        if tracked_val > 0:
            above = max(0.0, win.state.summary * 1.0 - MIN_TRACKED)
            above = above /(MAX_TRACKED - MIN_TRACKED)
            percent = min(1.0, above)
            #print "%", percent, "<--", win.state.summary
            pyautogui.moveTo(1200, BOTTOM_PIXEL - percent * (BOTTOM_PIXEL - TOP_PIXEL))

win = PlotWindow()
#CHUNK = 1024
CHUNK = 1024
CHANNELS = 1
#RATE = 44100
RATE = 11025
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)

q = Queue()

def reader(win, qout):
    print "making"
    p = pyaudio.PyAudio()
    print "made p", p
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    #output=True,
                    output=False,
                    frames_per_buffer=CHUNK)
    print "made stream", stream, dir(stream), stream._rate, stream._input_latency

    while True:
        data = stream.read(CHUNK)
        #stream.write(data, CHUNK)
        count = len(data)/2 # 2 bytes / sample
        format = "%dh"%(count)
        shorts = numpy.array(struct.unpack( format, data ))
        shorts = shorts * SHORT_NORMALIZE
        qout.put(copy.copy(shorts))
        if win.done: break

    stream.stop_stream()
    stream.close()

t = threading.Thread(
        name='timer',
        target=timed_redraw,
        args=(win, win.plot_area, q))
t.start()

r = threading.Thread(
        name='reader',
        target=reader,
        args=(win, q))
r.start()

win.connect("delete-event", Gtk.main_quit)
win.show_all()

Gdk.threads_init()
Gtk.main()
