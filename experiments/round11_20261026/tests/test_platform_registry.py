"""Availability follows the fixed reference strategy without extra file scans."""
from paper3r11.registry import SEEDS, validate_registry


def test_missing_optional_reuse_does_not_block_meta_train_reference_build(tmp_path):
    registry={'root':str(tmp_path),'seeds':list(SEEDS),'profile_reference_by_seed':{},
              'profile_raw_reference_by_seed':{},'missing_artifacts':[dict(seed=SEEDS[0],
              scope='development_reuse',kind='completed_R9_EA_feature_reference')]}
    result=validate_registry(registry,stage='development',seeds=[SEEDS[0]])
    assert result['status']=='READY' and result['reference_strategy']=='build_meta_train_prefix'
    registry['missing_artifacts'].append(dict(scope='development',kind='source_cache_manifest'))
    result=validate_registry(registry,stage='development',seeds=[SEEDS[0]])
    assert result['status']=='MISSING_INPUTS'
    assert [r['kind'] for r in result['missing_artifacts']]==['source_cache_manifest']


def test_unused_partial_reference_artifact_is_not_a_new_fallback_gate(tmp_path):
    registry={'root':str(tmp_path),'seeds':list(SEEDS),'profile_reference_by_seed':{
                str(SEEDS[0]):{'checkpoint':{'path':'unused-missing.pt','sha256':'a'*64,'bytes':10}}},
              'profile_raw_reference_by_seed':{},'missing_artifacts':[]}
    result=validate_registry(registry,seeds=[SEEDS[0]])
    assert result['status']=='READY' and result['reference_strategy']=='build_meta_train_prefix'
