from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import tensorflow as tf


class LinearDecay(LearningRateSchedule):
    def __init__(self, nsteps, initlr, shiftstep, finallr):
        super().__init__()
        self.nsteps = nsteps
        self.initlr = initlr
        self.shiftstep = shiftstep
        self.finallr = finallr

    def __call__(self, step):
        """
        return a float(learning rate)
        """

        # decrease linearly
        steprate = math_ops.abs(math_ops.divide(
            math_ops.subtract(self.finallr, self.initlr), self.nsteps))

        lr = math_ops.subtract(self.initlr, math_ops.multiply(
            steprate, math_ops.subtract(step, self.shiftstep)))

        pred = math_ops.greater(step, self.shiftstep)
        lr = control_flow_ops.cond(pred, lambda: lr, lambda: self.initlr)

        return lr
