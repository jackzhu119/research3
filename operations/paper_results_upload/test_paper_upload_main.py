import json
from pathlib import Path
import tempfile
import unittest

import paper_upload_main as main
from paper_results_collect import collect_round

class FakeGitHub:
    def __init__(self):self.files={}
    def git_publish(self, files, message):
        self.files.update(files)
        return 'a'*40

class UploadTests(unittest.TestCase):
    def test_reports_publish_as_text_without_touching_raw_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project';out=root/'runs/followup_20261023_v2';out.mkdir(parents=True)
            (out/'plan.json').write_text(json.dumps({'tasks':[{'key':'a'}]}))
            (out/'reports').mkdir();(out/'reports/report.json').write_text(json.dumps({'status':'COMPLETE','complete_tasks':1,'total_tasks':1}))
            (out/'reports/data.csv').write_text('participant,score\np1,0.6\n')
            work=Path(d)/'fast';full=Path(d)/'paper3-results-upload';full.mkdir();sentinel=full/'upload.log';sentinel.write_text('keep')
            captured=collect_round(root,2,work,include_tasks=False)
            gh=FakeGitHub();receipt=main.publish_round(gh,captured,work)
            self.assertEqual(receipt['status'],'PAPER_REPORT_FILES_UPLOADED')
            self.assertFalse(receipt['full_raw_results_uploaded'])
            self.assertTrue(any(p.endswith('/files/reports/data.csv') for p in gh.files))
            self.assertFalse(any(p.endswith('/LATEST.json') for p in gh.files))
            self.assertIn('results/paper3/round-02/PAPER_DATA_LATEST.json',gh.files)
            self.assertEqual(sentinel.read_text(),'keep')
            self.assertFalse((out/'a').exists())
    def test_plan_only_is_partial_paper_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project';out=root/'runs/followup_20261023_v2';out.mkdir(parents=True)
            (out/'plan.json').write_text(json.dumps({'tasks':[{'key':'a'}]}))
            work=Path(d)/'fast';captured=collect_round(root,2,work,include_tasks=False)
            gh=FakeGitHub();receipt=main.publish_round(gh,captured,work)
            self.assertEqual(receipt['status'],'PAPER_FILES_PARTIAL')
            self.assertEqual(receipt['report_files'],0)

if __name__=='__main__':unittest.main()
