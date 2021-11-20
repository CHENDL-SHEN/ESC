# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import json

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')

class DottableDict(dict): 
  def __init__(self, *args, **kwargs): 
    dict.__init__(self, *args, **kwargs) 
    self.__dict__ = self 
  def allowDotting(self, state=True): 
    if state: 
      self.__dict__ = self 
    else: 
      self.__dict__ = dict() 