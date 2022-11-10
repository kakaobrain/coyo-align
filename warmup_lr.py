import tensorflow as tf

class WarmUpExtension:
    def __init__(self, warmup_steps, *args, **kwargs):
        ws = kwargs.pop("warmup_steps", warmup_steps)
        super().__init__(*args, **kwargs)
        self.warmup_steps = ws

    @tf.function
    def __call__(self, step):

        step = tf.cast(step, tf.float32)
        if step < self.warmup_steps:
            lr = super().__call__(0)
            lr = lr * step / self.warmup_steps
        else:
            lr = super().__call__(step - self.warmup_steps)

        return lr

    def get_config(self):
        config = super().get_config()
        config.update(
            {"warmup_steps": self.warmup_steps, }
        )
        return config


def extend_with_warmup_lr(base_scheduler):

    class WarmupLrScheduler(WarmUpExtension, base_scheduler):
        def __init__(self, warmup_steps, *args, **kwargs):
            super().__init__(warmup_steps, *args, **kwargs)

    return WarmupLrScheduler
