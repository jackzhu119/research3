import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuation_delivery_test_builder',HERE/'build_delivery.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

@pytest.fixture
def package(tmp_path):
    root=tmp_path/'code';root.mkdir()
    for name in builder.REQUIRED:
        (root/name).write_bytes((HERE/name).read_bytes() if name in ('installer_template.py','build_delivery.py') else b'FIXTURE_CODE_ONLY=True\n')
    record=builder.build(root)
    installer=root/'delivery'/record['entry']
    namespace={'__name__':'delivery_software_fixture','__file__':str(installer)}
    exec(compile(installer.read_bytes(),str(installer),'exec'),namespace)
    return root,record,namespace

def test_deterministic_package(package):
    root,first,_=package
    assert builder.build(root)==first

def test_default_install_only_and_idempotent(package,tmp_path):
    _,record,ns=package
    home=tmp_path/'home';home.mkdir()
    folder=ns['install'](home)
    assert folder==ns['install'](home)
    assert folder.name==record['payload_sha256'][:20]
    assert (folder/'entry.py').is_file()

def test_existing_tamper_refused(package,tmp_path):
    _,_,ns=package
    home=tmp_path/'home';home.mkdir();folder=ns['install'](home)
    (folder/'entry.py').write_text('changed')
    with pytest.raises(ValueError,match='IMMUTABLE'):ns['install'](home)

def test_symlink_install_base_refused(package,tmp_path):
    _,_,ns=package
    home=tmp_path/'home';home.mkdir();outside=tmp_path/'other';outside.mkdir()
    (home/ns['BASE_NAME']).symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='SYMLINK'):ns['install'](home)

def test_archive_checksum_refused(package):
    _,_,ns=package
    ns['PAYLOAD_SHA256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):ns['checked_payload']()

def test_unexpected_installed_file_refused(package,tmp_path):
    _,_,ns=package
    home=tmp_path/'home';home.mkdir();folder=ns['install'](home)
    (folder/'unexpected.py').write_text('unexpected')
    with pytest.raises(ValueError,match='IMMUTABLE'):ns['install'](home)
