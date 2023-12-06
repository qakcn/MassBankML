# Copyright 2023 qakcn
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
## This script is useful to timing.                                           ##
##                                                                            ##
## Author: qakcn                                                              ##
## Email: qakcn@hotmail.com                                                   ##
## Version: 1.0                                                               ##
## Date: 2023-12-01                                                           ##
################################################################################


if __name__ == '__main__':
    raise SystemExit('This script is not meant to be run directly')

import time

class Timer:
    """A simple timer class"""
    timer = []
    named_timer={}

    @staticmethod
    def tick(name=None):
        """Start a new timer. When name is given, it is a named timer. A named timer won't stop even tock(name) is called. Call tick(name) again to reset the named timer.
        Parameters: name - a string for named timer, default to None.
        """
        t=time.perf_counter()
        if name is None:
            Timer.timer.append(t)
        else:
            Timer.named_timer[name]=t

    @staticmethod
    def tock(name=None):
        """Calculate the time elapsed since last tick() call. When name is given, it is a named timer. A named timer won't stop even tock(name) is called. Call tick(name) again to reset the named timer.
        Parameters: name - a string for named timer, default to None.
        """
        if name is None:
            b=Timer.timer.pop()
        else:
            b=Timer.named_timer[name]
        return time.perf_counter() - b