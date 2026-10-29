"""Real background installer with a retained user stop, without Slurm access."""
import json
import os
from pathlib import Path
import signal
import sys
import time

from test_live_hook import layout, hotfix, package_hashes


def configure(layout):
    root,home=layout['root'],layout['home']
    runtime=root/'runs/training_campaign_parallel_v1';runtime.mkdir(parents=True)
    single=home/'paper3-r1-single-gpu/work';single.mkdir()
    config=dict(root=str(root),home=str(home),runtime=str(runtime),native_package=str(layout['native']),
        python=sys.executable,worker_command=[sys.executable,'-I','-S','-B','-u',str(layout['package']/'worker_runner.py'),
        '--root',str(root),'--runtime-dir',str(runtime),'--slot','{slot}','--package',str(layout['native'])])
    (single/'config.json').write_text(json.dumps(config))
    (runtime/'STOP').write_text('explicit user cancellation retained\n')
    return runtime


def test_actual_detached_apply_preserves_user_stop_and_immutable_release(layout):
    module=hotfix();runtime=configure(layout);work=layout['work']
    before=(runtime/'STOP').read_bytes()
    result=module.apply(layout['root'],layout['home'],work)
    try:
        assert result['terminal_can_close'] is True
        assert result['slurm_jobs_cancelled']==result['slurm_jobs_submitted']==0
        deadline=time.monotonic()+5
        status={}
        while time.monotonic()<deadline:
            status=module.read(work/'status.json',{})
            if status.get('status')=='FIX_INSTALLED_CAMPAIGN_STOP_RETAINED':break
            time.sleep(.05)
        assert status['status']=='FIX_INSTALLED_CAMPAIGN_STOP_RETAINED'
        assert status['signal_sent'] is False
        assert (runtime/'STOP').read_bytes()==before
        assert package_hashes(layout['package'])==layout['before']
        assert layout['startup'].read_bytes().startswith(layout['original_startup'])
        assert os.getsid(result['watch_pid'])==result['watch_pid'] or module.identity(result['watch_pid']) is None
    finally:
        current=module.identity(result.get('watch_pid'))
        record=module.read(work/'watch-process.json',{})
        if current is not None and current.get('argv')==record.get('argv') and current.get('start_ticks')==record.get('start_ticks'):
            os.kill(current['pid'],signal.SIGTERM)


def test_context_refuses_different_root_without_modifying_startup(layout):
    module=hotfix();configure(layout)
    wrong=layout['home']/'another-project';wrong.mkdir()
    before=layout['startup'].read_bytes()
    try:
        module.apply(wrong,layout['home'],layout['work'])
    except ValueError as error:
        assert str(error)=='EXACT_SINGLE_GPU_CONFIGURATION_REQUIRED'
    else:
        raise AssertionError('Foreign root accepted')
    assert layout['startup'].read_bytes()==before
