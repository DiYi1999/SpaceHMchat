def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()


"""最后没用了，指定了EarlyStopping的check_on_train_epoch_end参数为True，就可以在训练结束时检查了"""
class MyEarlyStopping(pl.callbacks.EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)