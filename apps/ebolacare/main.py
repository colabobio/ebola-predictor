import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty, NumericProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.screenmanager import SlideTransition, RiseInTransition, FallOutTransition

from kivy.storage.jsonstore import JsonStore
from os.path import join

from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy import platform

import os
import re
import glob
import operator
import numpy as np
from utils import gen_predictor

####################################################################################
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

####################################################################################
# Load the kv file specifying the UI, otherwise needs to be named
# EbolaPredictor.kv (see note in App class below) and will be loaded
# automatically.
if platform == 'android':
    Builder.load_file('ui-android.kv')
elif platform == 'ios':
    Builder.load_file('ui-ios.kv')
else:
    Builder.load_file('ui.kv')
        
####################################################################################
# Load variables
values = {}
units = {}

variables = []
var_label = {}
var_kind = {}
var_def_unit = {}
var_alt_unit = {}
var_unit_conv = {}
with open("variables.csv", "r") as vfile:
    lines = vfile.readlines()
    for line in lines[1:]:
        line = line.strip()
        row = line.split(",")
        name = row[0]
        label = row[1]
        kind = row[2] 
        def_unit = row[3]
        alt_unit = row[4]
        unit_conv = row[5]
 
        variables.append(name)
        var_label[name] = label
        var_kind[name] = kind
        var_def_unit[name] = def_unit
        var_alt_unit[name] = alt_unit
        if unit_conv:
            c, f = unit_conv.split("*")
            unit_cf = [float(c), float(f)]
        else:
            unit_cf = [0, 1]
        var_unit_conv[name] = unit_cf

        units[name] = def_unit
        print name, label, kind, def_unit, alt_unit, unit_cf

####################################################################################
# Create the screen manager

# Base class for all input screens
class InputScreen(Screen):
    def clear_widgets(self):
        for widget in self.walk():
            if type(widget) == kivy.uix.textinput.TextInput:
                widget.text = ""
            elif type(widget) == kivy.uix.spinner.Spinner:
                if not ".unit" in widget.name:
                    widget.text = "Unknown"

    def set_units(self):
        print units 
        print "YEAHHHHH",self
        for widget in self.walk():
            if type(widget) == kivy.uix.spinner.Spinner and ".unit" in widget.name:
                var,_ = widget.name.split(".")
                print units[var]
                widget.text = units[var]
 
# Declare all screens
class InputScreenDemo(InputScreen):
    pass

class InputScreenChart(InputScreen):
    pass

class InputScreenLab(InputScreen):
    pass

class InputScreenPCR(InputScreen):
    pass

class ResultScreen(Screen):
    curr_risk_color = ListProperty([0, 0, 0, 1])
    curr_risk_label = StringProperty('NONE')
    curr_risk_level = NumericProperty(0)
    pass

class ImageButton(ButtonBehavior, Image):
    pass

in_scr = [None] * 4
in_scr[0] = InputScreenDemo(name='input 1')
in_scr[1] = InputScreenChart(name='input 2')
in_scr[2] = InputScreenLab(name='input 3')
in_scr[3] = InputScreenPCR(name='input 4')
res_scr = ResultScreen(name='result')
sm = ScreenManager()
for iscr in in_scr: sm.add_widget(iscr)
sm.add_widget(res_scr)

####################################################################################
# Read the predictive models

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
    d = pair[0]
    v = []
    with open(os.path.join(d, "variables.txt")) as vfile:
        lines = vfile.readlines()
        for line in lines[1:]:
            line = line.strip()
            parts = line.split()
            v.append(parts[0])
    info = [d, v] 
    #print info
    models_info.append(info)

####################################################################################
# Main app

class EbolaCAREApp(App):
    def build(self):
        self.categories = {"Unknown":"", "Yes":"1", "No":"0", "Female":"1", "Male":"0"}
        data_dir = getattr(self, 'user_data_dir') #get a writable path to save the file
        self.store = JsonStore(join(data_dir, 'units.json'))
        print "DATA DIR", data_dir
        # Load last used units
        for var in units:
            if self.store.exists(var):
                units[var] = self.store.get(var)['unit']
        for scr in in_scr: scr.set_units()
        return sm

    def on_pause(self):
        # Here you can save data if needed
        return True

    def on_resume(self):
        # Here you can check if any data needs replacing (usually nothing)
        pass

    def set_var_value(self, name, value):
        if value in self.categories:
            value = self.categories[value]
        values[name] = value    
        print name, value

    def set_var_unit(self, name, unit):
        units[name] = unit
        self.store.put(name, unit=unit)
        print name, unit

    def restart(self):
        values = {}
        for scr in in_scr: scr.clear_widgets()
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
        if "AGE" in values and values["AGE"]:
            age = 30
            try:                  
                age = float(values["AGE"])
            except ValueError:
                pass
            except TypeError:
                pass
            if age < 10 or 50 < age: 
                res_scr.curr_risk_color = [153.0/255, 93.0/255, 77.0/255, 1]
                res_scr.curr_risk_label = 'HIGH RISK'            
                res_scr.curr_risk_level = 1
                sm.current = 'result'
                return        

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
            #print res, info[1]
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
                    if var in units and units[var] != var_def_unit[var]:
                        # Need to convert units
                        c, f = var_unit_conv[var]
                        print "convert",var,v[i + 1],"->",f*(v[i + 1] + c)
                        v[i + 1] = f * (v[i + 1] + c)
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
    EbolaCAREApp().run()

