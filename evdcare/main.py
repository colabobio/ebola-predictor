import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty, NumericProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.screenmanager import SlideTransition, RiseInTransition, FallOutTransition

from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior

import os
import re
import glob
import operator
import numpy as np
from utils import gen_predictor

#Fonts
from kivy.core.text import LabelBase
KIVY_FONTS = [
    {
        "name": "LatoRegular",
        "fn_regular": "./fonts/LatoRegular.ttf",
    }
]
for font in KIVY_FONTS:
    LabelBase.register(**font)

# Load the kv file specifying the UI, otherwise needs to be named
# EbolaPredictor.kv (see note in App class below) and will be loaded
# automatically.
Builder.load_file('ui.kv')

# Load variables
variables = []
var_label = {}
var_unit = {}
var_kind = {}
with open("variables.csv", "r") as vfile:
    lines = vfile.readlines()
    for line in lines[1:]:
        line = line.strip()
        row = line.split(",")
        name = row[0]
        label = row[1]
        unit = row[2]
        kind = row[3] 
        variables.append(name)
        var_label[name] = label
        var_unit[name] = unit
        var_kind[name] = kind
values = {}

class InputScreen(Screen):
    def clear_widgets(self):
        for widget in self.walk():
            if type(widget) == kivy.uix.textinput.TextInput:
                widget.text = ""
            elif type(widget) == kivy.uix.checkbox.CheckBox:
                if "_na" in widget.name:
                    widget.active = True
                else:
                    widget.active = False

# Declare both screens
class InputScreen1(InputScreen):
    pass

class InputScreen2(InputScreen):
    pass

class InputScreen3(InputScreen):
    pass

class ResultScreen(Screen):
    curr_risk_color = ListProperty([0, 0, 0, 1])
    curr_risk_label = StringProperty('NONE')
    curr_risk_level = NumericProperty(0)
    pass

class ImageButton(ButtonBehavior, Image):
    pass

# Create the screen manager
in_scr = [None] * 3
in_scr[0] = InputScreen1(name='input 1')
in_scr[1] = InputScreen2(name='input 2')
in_scr[2] = InputScreen3(name='input 3')
res_scr = ResultScreen(name='result')
sm = ScreenManager()
for iscr in in_scr: sm.add_widget(iscr)
sm.add_widget(res_scr)

ranking = {}
dirs = glob.glob("models/*")
for d in dirs:
    with open(os.path.join(d, "ranking.txt")) as rfile:
        line = rfile.readlines()[0].strip()
        f1score = float(line.split()[0])
        ranking[d] = f1score 

sorted_ranking = reversed(sorted(ranking.items(), key=operator.itemgetter(1)))
models_info = []
for pair in sorted_ranking:
#    print , pair[1]
    d = pair[0]
    v = []
    with open(os.path.join(d, "variables.txt")) as vfile:
        lines = vfile.readlines()
        for line in lines[1:]:
            line = line.strip()
            parts = line.split()
            v.append(parts[0])
    info = [d, v] 
    print info
    models_info.append(info)

class EbolaPredictorApp(App):
    def build(self):
        return sm

    def on_pause(self):
        # Here you can save data if needed
        return True

    def on_resume(self):
        # Here you can check if any data needs replacing (usually nothing)
        pass

    def set_var_value(self, name, value):
        values[name] = value    
        print name, value

    def restart(self):
        values = {}
        print "***********************"
        in_scr[0].clear_widgets()
        in_scr[1].clear_widgets()
        in_scr[2].clear_widgets()
        print "***********************"
        sm.transition = FallOutTransition()
        sm.current = 'input 1'
  
    def go_screen(self, scr):
        curr_scr = int(sm.current.split()[1])
        if curr_scr < scr:
            sm.transition = SlideTransition(direction='left')
        else:
            sm.transition = SlideTransition(direction='right')
        sm.current = 'input ' + str(scr)

    def calc_risk(self):
        # Find highest ranking model that contained in the provided variables
        model_dir = None
        model_vars = None
        vv = set([])
        for k in values:
            if values[k]: vv.add(k)
        print vv 
        for info in models_info: 
            v = set(info[1])
            res = v.issubset(vv)
            print res, info[1]
            if res:
                model_dir = info[0]
                model_vars = info[1]
                break    
        
        if not model_dir or not models_info:
            res_scr.curr_risk_color = [0.5, 0.5, 0.5, 1]
            res_scr.curr_risk_label = 'INSUFFICIENT DATA'            
            res_scr.curr_risk_level = 0
            sm.transition = RiseInTransition()
            sm.current = 'result'
            return

        print "FOUND MODEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"     
        print model_dir
        print model_vars
            
        predictor = gen_predictor(os.path.join(model_dir, 'nnet-params'))
        N = len(model_vars)
        
        model_min = []
        model_max = []
        with open(os.path.join(model_dir, 'bounds.txt')) as bfile:
            lines = bfile.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split()
                model_min.append(float(parts[1]))  
                model_max.append(float(parts[2]))

        print model_min
        print model_max 

        v = [None] * (N + 1)
        v[0] = 1
        for i in range(N):
            var = model_vars[i]
            if var in values:
                try:
                    v[i + 1] = float(values[var])
                except ValueError:
                    pass
                except TypeError:
                    pass

        if None in v:
            res_scr.curr_risk_color = [0.5, 0.5, 0.5, 1]
            res_scr.curr_risk_label = 'INSUFFICIENT DATA'            
            res_scr.curr_risk_level = 0
            sm.current = 'result'
            return

        for i in range(N):
            f = (v[i + 1] - model_min[i]) / (model_max[i] - model_min[i]) 
            if f < 0: v[i + 1] = 0
            elif 1 < f: v[i + 1] = 1
            else: v[i + 1] = f

        print values
        print v

        X = np.array([v])
        probs = predictor(X)
        pred = probs[0]
        print "------------->",pred,type(pred)
        res_scr.curr_risk_level = float(pred)
        if pred < 0.5:
            res_scr.curr_risk_color = [121.0/255, 192.0/255, 119.0/255, 1]
            res_scr.curr_risk_label = 'LOW RISK'
        else:
            level = float((pred - 0.5) / 0.5)
            res_scr.curr_risk_color = [153.0/255, 93.0/255, 77.0/255, 1]
            res_scr.curr_risk_label = 'HIGH RISK' 
        sm.transition = RiseInTransition() 
        #sm.transition = SlideTransition(direction='left')
        sm.current = 'result'
 
if __name__ == '__main__':
    EbolaPredictorApp().run()

