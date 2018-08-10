import datetime

class Timer(object):
    def __init__(self, title):
        self.title = title

    def start(self):
        """Starts the timer"""
        self.start_time = datetime.datetime.now()
        print("[%s] Starting %s" %(self.start_time, self.title))
        return self
    
    def time_elapsed(self):
        """Prints and returns the time elapsed"""
        self.stop_time = datetime.datetime.now()
        time_elapsed = (self.stop_time-self.start_time).seconds/60.0
        print("\r[%s] Done with %s (Took %.3f minutes)" %(datetime.datetime.now(), self.title, time_elapsed))
        return time_elapsed    