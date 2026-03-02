import os
from wm.registry import ModelRegistry
from wm.cfg import RegCfg

def test_push_pull(tmp_dir):
    src=os.path.join(tmp_dir,"model_src")
    os.makedirs(src)
    open(os.path.join(src,"w.bin"),"w").write("fake")
    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v=reg.push(src,{"loss":0.5})
    assert v.vid
    got=reg.pull(v.vid)
    assert got.vid==v.vid
    assert got.metrics["loss"]==0.5
    reg.close()

def test_latest(tmp_dir):
    src=os.path.join(tmp_dir,"model_src")
    os.makedirs(src)
    open(os.path.join(src,"w.bin"),"w").write("fake")
    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v1=reg.push(src,{"loss":1.0})
    v2=reg.push(src,{"loss":0.5},parent=v1.vid)
    l=reg.latest()
    assert l.vid==v2.vid
    reg.close()

def test_list_versions(tmp_dir):
    src=os.path.join(tmp_dir,"model_src")
    os.makedirs(src)
    open(os.path.join(src,"w.bin"),"w").write("fake")
    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    reg.push(src,{"loss":1.0})
    reg.push(src,{"loss":0.5})
    vs=reg.list_versions()
    assert len(vs)==2
    reg.close()

def test_rollback(tmp_dir):
    src=os.path.join(tmp_dir,"model_src")
    os.makedirs(src)
    open(os.path.join(src,"w.bin"),"w").write("fake")
    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v1=reg.push(src,{"loss":1.0})
    reg.push(src,{"loss":0.8})
    v3=reg.rollback(v1.vid)
    assert v3.parent==v1.vid
    assert reg.latest().vid==v3.vid
    reg.close()

def test_push_single_file(tmp_dir):
    f=os.path.join(tmp_dir,"model.bin")
    open(f,"w").write("weights")
    reg=ModelRegistry(RegCfg(local_dir=os.path.join(tmp_dir,"reg")))
    v=reg.push(f,{"loss":0.3})
    assert v.vid
    assert os.path.isdir(v.path)
    reg.close()
