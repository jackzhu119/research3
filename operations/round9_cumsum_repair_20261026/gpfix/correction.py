"""One exact R9 source correction; no native code file is written here."""
import hashlib

ORIGINAL_GP_SHA256 = '5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7'
OLD_CUMSUM = '        cumulative=torch.cumsum(pairweights[order],0)\n'
NEW_CUMSUM = ('        cumulative=torch.cumsum(pairweights[order].detach().cpu(),0)'
              '.to(pairweights.device)\n')


def corrected_source(fullname, source):
    if fullname != 'paper3r9.gp':
        return source, []
    if source.count(NEW_CUMSUM) == 1 and source.replace(NEW_CUMSUM, '', 1).count(OLD_CUMSUM) == 0:
        return source, []
    if source.count(OLD_CUMSUM) != 1 or source.count(NEW_CUMSUM) != 0:
        raise RuntimeError('Unexpected R9 weighted-reference cumsum source span')
    fixed = source.replace(OLD_CUMSUM, NEW_CUMSUM, 1)
    compile(fixed, 'paper3r9/gp.py', 'exec')
    return fixed, ['weighted_reference_deterministic_cpu_cumsum']


def evidence(source):
    fixed, labels = corrected_source('paper3r9.gp', source)
    return {'module':'paper3r9.gp',
            'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'executed_source_sha256':hashlib.sha256(fixed.encode()).hexdigest(),
            'corrections':labels,'native_source_file_changed':False,
            'deterministic_algorithms_disabled':False,
            'operation_device':'CPU float64 prefix sum; restore original weights device',
            'max_pairweights':256*255//2,
            'max_transfer_bytes_each_direction':8*(256*255//2)}
