"""Amendment 3, deviation 4: rerun the frozen loader and confirmatory script with CoderData's fit_ic50 on the scale the
registered orientation rule assumed (protocol/deviations.md, item 4).

CoderData stores fit_ic50 as -log10(IC50 / M) (a pIC50; coderbuild/utils/fit_curve.py), whereas the registered rule
compares fit_auc with an IC50 that increases with resistance. The rule's Spearman correlation against fit_ic50 is
therefore negated (Spearman(x, log10 IC50) = -Spearman(x, pIC50)); nothing else changes. The frozen 71_load_pdm.py
and 72_pdm_confirm.py are imported and run unchanged, writing to data/processed/pdm_corrected/ and
results/confirmation_PDM_corrected/.
"""
import importlib.util
import sys
from pathlib import Path

from common import PROCESSED, RESULTS

HERE = Path(__file__).resolve().parent


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def main():
    L = _module('pdm_loader_corrected', '71_load_pdm.py')
    registered = L.median_spearman

    def median_spearman(w, a, b):
        rho = registered(w, a, b)
        return -rho if (b == 'fit_ic50' and rho is not None) else rho

    L.median_spearman = median_spearman
    L.OUT = PROCESSED / 'pdm_corrected'
    L.main()
    C = _module('pdm_confirm_corrected', '72_pdm_confirm.py')
    C.main(base=PROCESSED / 'pdm_corrected', out=RESULTS / 'confirmation_PDM_corrected')


if __name__ == '__main__':
    main()
