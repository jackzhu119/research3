import copy
from collections import Counter
from pathlib import Path
import tempfile

from paper3r8.plan import build, SEEDS
from paper3r8.transport import TransportConfig


def test_finite_complete_graph_and_resource_contract():
    plan=build('/public/home/zhuziyuan/projects/paper3')
    keys={t['key'] for t in plan['tasks']}
    assert len(keys)==len(plan['tasks'])==2835
    assert Counter(t['family'] for t in plan['tasks'])['train']==24
    assert plan['resources']['partition']=='hpc_gpu'
    assert plan['resources']['jobs']==3
    assert plan['resources']['cpus_per_job']==16
    assert plan['resources']['memory_per_job']=='64G'
    assert {t['seed'] for t in plan['tasks']}==set(SEEDS)
    for t in plan['tasks']:
        TransportConfig(**t['config'])
        assert set(t['requires'])<=keys
        assert t['key'] not in t['requires']
        if t.get('meta_key'): assert t['meta_key'] in t['requires']
        if t.get('reference_key'): assert t['reference_key'] in t['requires']
        if t.get('external_manifest'): assert Path(t['external_manifest']).is_file()


def test_recipe_portable_across_roots_and_ignores_resource_location():
    a=build('/tmp/location-a')
    b=build('/tmp/location-b')
    assert a['recipe_sha256']==b['recipe_sha256']
    assert a['output']!=b['output']


def test_primary_family_excludes_grid_winner_selection():
    p=build('/tmp/root')
    core={t['group'] for t in p['tasks'] if t['family']=='core'}
    for a,b in p['comparisons']:
        assert a in core and b in core
        assert 'grid_' not in a+b
    for t in p['tasks']:
        if t['family']=='label_null':
            assert t['config']['covariance_weighting']=='pooled'
            assert t['scenarios']==['support_label_permutation']
