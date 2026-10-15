"""Verifier regressions using the standard library; no installs or model execution."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_inventory', ROOT / 'tools/verify_inventory.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ArtifactTests(unittest.TestCase):
    def test_same_size_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'datasets/example.bin'
            path.parent.mkdir()
            path.write_bytes(b'good')
            manifest = {'groups': {'datasets': {'files': [{
                'path': 'datasets/example.bin', 'bytes': 4,
                'sha256': hashlib.sha256(b'good').hexdigest(),
            }]}}}
            results, errors = verify.verify_artifacts(root, manifest)
            self.assertEqual(results['datasets']['verified'], 1)
            self.assertFalse(errors)
            path.write_bytes(b'evil')
            results, errors = verify.verify_artifacts(root, manifest)
            self.assertEqual(results['datasets']['verified'], 0)
            self.assertIn('checksum/size mismatch', errors[0])

    def test_path_traversal_is_rejected(self):
        for path in ('../outside', '/absolute/path', 'datasets/../../outside', 'datasets\\outside'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify.relative_path(path)


class CheckoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / 'checkout'
        # Copy only the shipped source inventory, never research artifacts.
        files = json.loads((ROOT / 'manifests/source_files.json').read_text())['files']
        for name in files:
            target = cls.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_check(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = verify.main(['--root', str(self.root), *args])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_source_only_passes_and_strict_scope_fails(self):
        result, output, error = self.run_check()
        self.assertEqual(result, 0, error)
        self.assertIn('137 missing', output)
        self.assertIn('omitted artifacts are not verified', output)
        for scope in ('datasets', 'saved-results', 'all'):
            result, output, error = self.run_check('--require', scope)
            self.assertEqual(result, 1)
            self.assertIn('Required datasets incomplete', error)
            self.assertIn('ARTIFACTS.md', error)

    def test_present_corruption_fails_default_mode(self):
        manifest = json.loads((self.root / 'manifests/artifacts.json').read_text())
        name = manifest['groups']['evaluations']['files'][0]['path']
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(b'broken')
            result, output, error = self.run_check()
            self.assertEqual(result, 1)
            self.assertIn(name, error)
        finally:
            path.unlink()

    def test_missing_configuration_fails_source_check(self):
        name = 'models/Burgers/fluxgraphnet_with_advection/config_release.json'
        path = self.root / name
        data = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing repository file'):
                self.run_check()
        finally:
            path.write_bytes(data)


if __name__ == '__main__':
    unittest.main()
