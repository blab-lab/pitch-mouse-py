from gi.repository import Gtk, GObject, Gdk
from runmlp import MLP, LinearOutputLayer, HiddenLayer
import traceback
from pitch import PitchFinder, FormantFinder, FormantFinderNeural
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
from scipy.signal import lfilter
import pyautogui


import sys


finders = {
  'lpc': FormantFinder,   
  'neural': FormantFinderNeural,        
  'pitch': PitchFinder,        
}

finder_name = 'lpc'
if len(sys.argv) >= 2:
    finder_name = sys.argv[1]
finder = finders[finder_name]

CHUNKS_IN_VIEW = 200
FRAME_RATE_MS = 60.0
SUMMARY_BUFFER_MS = 50

class State(object):
    pass

class PlotWindow(Gtk.Window):

    def __init__(self):
        title = "Sound tracker: %s"%finder_name
        Gtk.Window.__init__(self, title=title)
        self.connect("delete-event", self.on_delete)
        self.done = False

        hb = Gtk.HeaderBar()
        hb.props.title = title
        self.set_titlebar(hb)
        self.state = State()
        self.chunk_num = 0
        self.state.summary = 0
        self.state.history = numpy.zeros((CHUNKS_IN_VIEW, 2))
        self.delta = .1

        self.box = Gtk.Box(spacing=6)
        self.add(self.box)

        self.plot_area = Gtk.DrawingArea()
        self.plot_area.connect("draw", self.draw_plot)

        self.plot_area.set_size_request(100,100)
        self.box.pack_start(self.plot_area, True, True, 0)
        self.color1 = (.3,.3,.3)
        self.color2 = (.6,.6,.6)
        self.color3 = (.8,.8,.8)

    def on_delete(self, *args):
        #print "Got delete", args
        self.done = True
        return False

    def draw_plot(self, widget, cr):
        t0 = time.time()

        oscope_width_px = 1000
        oscope_duration_ms = 4000 # ms
        oscope_height_px = 600

        slice_width_px = oscope_width_px / CHUNKS_IN_VIEW

        white = (255, 255, 255)
        cr.rectangle(0, 100, 2*oscope_width_px, oscope_height_px)
        cr.set_source_rgb(*white)
        cr.fill()

        HISTORY_POINTS = 20
        dot_plot = [(self.state.chunk_num-x) %  CHUNKS_IN_VIEW for x in range(HISTORY_POINTS, 0, -1)]
        colors  = [[1 - i * 1.0 / HISTORY_POINTS] * 3 for i in range(0, HISTORY_POINTS)]
        for ((i,c), color) in zip(enumerate(dot_plot), colors):
            c = self.state.history[c]
            print "2d", color, i,  c
            if c == None: continue
            dot_x_pos =  2*oscope_width_px - c[0] / 2
            dot_y_pos =  100 + oscope_height_px - c[1] / 5
            print "x,y", dot_x_pos, dot_y_pos, c
            cr.new_sub_path()
            cr.arc(dot_x_pos, dot_y_pos, slice_width_px*2, 0, 2*math.pi)
            cr.close_path()
            cr.set_source_rgb(*color)
            cr.fill()

        
        for i,c in enumerate(self.state.history[0:CHUNKS_IN_VIEW]):
            if c == None: continue
            dot_y_pos =  100 + oscope_height_px - c[0] / 6
            cr.new_sub_path()
            cr.arc(i * slice_width_px, dot_y_pos, slice_width_px/2, 0, 2*math.pi)
            cr.close_path()
        cr.set_source_rgb(*self.color2)
        cr.fill()

        for i,c in enumerate(self.state.history[0:CHUNKS_IN_VIEW]):
            if c == None: continue
            dot_y_pos =  100 + oscope_height_px - c[1] / 6
            cr.new_sub_path()
            cr.arc(i * slice_width_px, dot_y_pos, slice_width_px/2, 0, 2*math.pi)
            cr.close_path()
        cr.set_source_rgb(*self.color3)
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
    vals = np.zeros(128)
    print "vals len", len(vals)
    CHUNK_NUM  = 0
    deadline = time.time() + FRAME_RATE_MS * 1.0/ 1000
    dropped = False
    while True:
        sleepfor =  (deadline - time.time())
        time.sleep(sleepfor)
        if win.done: break

        while not q.empty():
            val =  q.get(False)
            if dropped: continue
            tracks = numpy.array([0,0])
            count = 0
            
            for i in range(8):
                tracked_val = finder.analyze(val[i*128:(i+1)*128], RATE)
                if tracked_val != None:
                    tracks += tracked_val
                    count += 1
            if numpy.any(tracks != [0,0]):
                mean = tracks / count * 10
                mean += (win.state.history[(CHUNK_NUM -1) % CHUNKS_IN_VIEW] / 1 * 5)
                mean += win.state.history[(CHUNK_NUM -2) % CHUNKS_IN_VIEW] / 1 *3
                mean += win.state.history[(CHUNK_NUM -4) % CHUNKS_IN_VIEW] / 1 * 2
                win.state.history[CHUNK_NUM % CHUNKS_IN_VIEW] = mean / 20
                CHUNK_NUM += 1
                win.state.chunk_num = CHUNK_NUM
                win.state.summary = numpy.mean(mean)

        Gdk.threads_enter()
        plot.queue_draw()
        Gdk.threads_leave()

        deadline = deadline + FRAME_RATE_MS * 1.0 / 1000
        dropped = False
        while deadline < time.time() + .002:
            deadline += FRAME_RATE_MS * 1.0 / 1000
            print "drop frame"
            dropped = True


        """
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
        """

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
