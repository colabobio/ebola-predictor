import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty
from kivy.uix.screenmanager import SlideTransition

import re
from utils import gen_predictor
import numpy as np

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
            elif type(widget) == kivy.uix.togglebutton.ToggleButton:
                widget.state = "normal"
#             print("{} -> {}".format(widget, widget.id))

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
#curr_scr = 1
#print curr_scr
#print "************************"

predictor = gen_predictor('params/nnet-params-0')

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
        sm.current = 'input 1'

    def go_screen(self, scr):
        curr_scr = int(sm.current.split()[1])
        if curr_scr < scr:
            sm.transition = SlideTransition(direction='left')
        else:
            sm.transition = SlideTransition(direction='right')
        sm.current = 'input ' + str(scr)

    def calc_risk(self):
        model_vars = ["PCR", "TEMP", "AST_1"]
        model_min = [0, 36, 0]
        model_max = [10, 40, 2000]
        N = len(model_vars)         

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

        print values
        print v

        if None in v:
            res_scr.curr_risk_color = [0.5, 0.5, 0.5, 1]
            res_scr.curr_risk_label = 'INSUFFICIENT DATA'            
            sm.current = 'result'
            return 

        for i in range(N): 
            v[i + 1] = (v[i + 1] - model_min[i]) / (model_max[i] - model_min[i])

        X = np.array([v])
        probs = predictor(X)

        pred = probs[0]
        if pred < 0.5:
            res_scr.curr_risk_color = [121.0/255, 192.0/255, 119.0/255, 1]
            res_scr.curr_risk_label = 'LOW RISK'
        else:
            res_scr.curr_risk_color = [153.0/255, 93.0/255, 77.0/255, 1]
            res_scr.curr_risk_label = 'HIGH RISK' 
        
        sm.transition = SlideTransition(direction='left')
        sm.current = 'result'
 
if __name__ == '__main__':
    EbolaPredictorApp().run()

