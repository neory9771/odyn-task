"""Judge the completed SFT1 test while SFT2 training continues on the remote box."""
import subprocess
import sys
import time
from pathlib import Path

from odyn_sft.common.storage import read_json

ROOT = Path('/dev/shm/odyn-e2e')


def main():
    report = ROOT / 'runs/qwen-sft1/train/final_evaluation/report.json'
    deadline = time.monotonic() + 3600
    while not report.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('SFT1 final evaluation did not finish')
        time.sleep(15)
    assert read_json(report)['status'] == 'complete'
    assert read_json(ROOT / 'runs/analysis-judge-controls/report.json')['passed']
    subprocess.run([sys.executable, '-u', '-m', 'odyn_sft.survey_evaluation.analysis_judge',
                    '--package', str(ROOT / 'work/package-variants'),
                    '--cases', str(ROOT / 'runs/qwen-sft1/train/final_evaluation/test/selected-final'),
                    '--out', str(ROOT / 'runs/sft1-judge'), '--workers', '8',
                    '--credentials-file', str(ROOT / 'private/credentials.json')], check=True)


if __name__ == '__main__':
    main()
