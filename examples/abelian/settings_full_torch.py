from types import SimpleNamespace
import yamps.tensor.backend_torch as back

settings_full_torch= SimpleNamespace(back= back, dot_merge=False, \
    sym= [], nsym=0, dtype='float64')