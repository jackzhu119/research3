#!/usr/bin/env python3
"""Install the R9 CUDA cumsum correction through its operational backend only."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

HOOK_B85 = 'c-qxiTW{R95`N!b!59c61+wb2r$y~z9~{R{jrxXNyG7w+5L}77%SN}7l$||+|9fXhN}~1>H#mLiKG+gD91g#kZ)T{yy}e1*lo>fBX;suri<!uUUcsA`Jk_Eqm6;Ja&q!6~D<Vpwmz+FR@*|g|mQ~7?N=Bn;DtM{IT%<%*jZ8V2!y-i(XjkkJmh(?Fm!beEA~KZK0_K%wD>7SYzCW+zK0d$}nx#4!UE_XFTr8HFXZr=u1S?6}6ip6>K5|lU%`&DL5!T2?8zYT<CpULLk)q0Y4t8iQ%EbsRsU>JJ=YkbrLfP<k1<Ug)WvJrA%^i6XdI`2k5GEPdTo$4fsvx{%Gr%Gl?d|Q2L{V2#lO<EjT+G^6`ws>ZHcK}L3zjZL$>pk@`a@M^`&_B^Nv+gqH2U>)GCjY(B4ZNN4B-22UmorQ1ga_a=i<};<F~=cm|E4m3@Js-DUC@WXF)`mA{5TRS#dNX@ca4t`WHI=`S{!Ke*lfn>_jaY41`WHfj?OoMahzX%EW>z9Y&+k^~HPo?)c*P>g4o2y||uE(V`&8_?*mG&dQW$G_RBj^J*cqipeUnTUz%i5Vtc)$)6}MYP||`uy_?kMhtvUQjD$eAb#tX;;&%pBdlGK@tCCDsF;(o(qvXud597TD_3B0zYKqp3B}2;EN}R!l$8vFcNGBjX9LiaG-pa-bV^Pn|FaPg7eUmmcUIq{@(0*I`JQ}ZY)Jt4(8@`!8wm!tCwxQBl2if)6k=Oe^{U4r2H~tRVK)xSs^kISYc&CgqpQ>3UIf<|Ox<1~Xw3Ag=I+Q4iyx2>974+;4V;_LO<1E>&~()@7vdha<}y&ohEyo9D#2(Tx=U9wT_(P#+@53e7hd?INcnh`*oS|k9}``L-kAn|bxVPRUlZiJ{pG@cbEAJCfRLRhcbC(<ON^ZM@O`EnTW8w55!Ot|7MMzATt37H(SJ)DhyRZnhy+O7OTaa)n=k^c2(TyDh6-+GadmYD0JB9SoHu!1vI64E4mgBS3F7;?bFIsV@-VTbC^#bbVKDGh5R;9(6_X9NMGx^7t&7@fZ0irT^@sMwK#P>ZL8-;VZD#SayV&Pe{d2P5+B@I<gAD^#*SUbIGlr3lLphuAJSJ71#RM`YGRi=3<a5U&l%!C>bqG=svg@s_xOPhNI%j|?OhQcn6W>No95fs8gCW`4qPTM#v+fmc16FPt@M=S&rSAkJp!7Qe-<8-4pxa4lOZjVP1Zt@6c=p#SAg@X2AZP)LdRDvsV!VFJ7A-txC+y%&;6E=jt98IXt0Zm(Auw$QJ!wPpvdWsAUsMcYJls;plc~{gCR!mEvl!pta+*RH5-Oh0y~FevN%n*%foY5niv5fcPM{lG_Q&Xm5|ebvN~p`Q>I>`vwZJyBb<pVf{*QqVAnJ$qfx>DoYCRy9p}`}USQmf-t^<MaDGUxU)bA2hQY=Kta?^OgRA+fS)MxFZw$%W+wcZ1)9W)$ktNACMHvKVos_F1ZcxjFveMzg=<AN%ODz;8GR_U^;KK5IAz0;Z=8^Wz^rUSFeGFW!Ta>dscm^4aRj-A$u+K7U~F$@*cYQs7!6d>`T6HF}jYE99}-Mfq9DgEtw^2_w*_~ewHTwk4?e}Ltc0tg<3gwl>#XQ!lx$#eHqOX$EuY)g}@DQXpd4)`O~Bs6PsNGEq!x969rVlp{BxrN|>#Pa6NdQd$NjUD_HoS1&(D>c5A4TsKRo;PY~UwUR(emYD-j{$v@+1P;nGe}%iluNi_Xs4$mA&O9opyQ!^@mQ!DNc&K@d{HOl+0&8*uMpw-d3!;-Frm~I6WRt;A8?d{7Prn%wcSdj7YMx?Z>e*`^lCurh88s@lRjXq_x+;Qo0)OUh`<!+$}mwet!22V0n;y94q#tF2NL;qd#d0rhSOLc`cZ2}n^a7haCznBV$ohuKiqh?Qt||M8|)oL2^Yv5Z>KX3#iP5JT2=)c-n$QatEis=usystj_n1ahi<teIovTp&bbL*Mjo1T0s|CpV-&8%Kp5;qLK~)!*uJhcwt!m*JVk+I#uP`N<G@_ukXaZ-rk4&RS52-Hq-4-%sP3I3QuZy{N;+%_pwUOH&?aL|Y{GcG+s2=nZo}NkZbpI{Zv?)_+8(a~0;6@3SU|S`w_b-~$1_n5OEBE8lS>HP8`!~4cJ1Ph@FiW~89B`)L?)Kfxvc)jO9f#XZZIw$wuO<GTIR4TRnW_<75JFosmI8K*zICoqCo@KO1nt5dCl3hjpl|~ea?by!SZb#Ne@WUTR60>HEY0Qqtaurv8#_8oKAAn<HY1EI-!?N;dH`bwl`hgr2wUE)3H$~aL!#TTviAr<_;Ta<IabHvkK1Z%@1;I;P+?i*U#If`QFrp6ug-)W7xa8WaVgRj%7M~<7E^u58&$csBYbE?!->U%i|BN#ZDy>8l8iOY*V)<ZuWcl8=3s`bF`kD$3u#2|M^F;CPnLz4YT;U^$6)dH+J~B#O-bZmUG`gwjZSMTEf+_YLKdbySHgE>()A&g~8E(4i0}f`VpCZ^cEjH>wES0J5^(`=#-4SPZ{_DqxFZ-yy#NL76-J2)I;3hit9S}gDz?+f!C|};|YX1mpVL%Hxp&lLifdX?)~ZQ>E!bK>U?^8e&X(|baHz9{&$3D^CGY#D!#UD_8al;3q$+CgxU`UXd&FN(T+u4!QI}weHSzn%6yC6-eteoZ|`ifpW~PNQ<ri+Z@nlvD&uv?>=f=Id-Oc|3y@rGyZ'
HOOK_SHA256 = '063cb6f1a178b9ed4f754103d15637f66d8e2937d4eb99855a9b034086c26c3f'
BASE_BACKEND_SHA256 = '576118c37439261d4d721ea12daf3416d06b96d83101e46c5a302a9aef83a64b'
OLD_MARKER = 'PAPER3_R589_RUNTIME_FIX_V1'
NEW_MARKER = 'PAPER3_R9_CUMSUM_RUNTIME_FIX_V2'


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary=path.with_name('.'+path.name+'.r9-fix-'+str(os.getpid()))
    try:
        with temporary.open('wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,mode);os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path,value):
    atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def wrappers(home,root):
    """Use the existing R9 launch and shallow R9 package roots only."""
    output=Path(root)/'runs/followup_20261025_r9'
    found=set()
    def add(path):
        path=Path(path).expanduser()
        if path.is_file():found.add(path.resolve())
    for relative in ('runtime/three_jobs/launch.json','runtime/launch.json'):
        launch=read(output/relative)
        for value in launch.get('worker_command',[]):
            if isinstance(value,str) and Path(value).name=='three_jobs.py':
                add(Path(value).parent/'backend_entry.py')
        for key in ('package','scientific_package','source_package'):
            if isinstance(launch.get(key),str):
                add(Path(launch[key])/'runtime/backend_entry.py')
    for pattern in ('paper3-round9-thin/*/runtime/backend_entry.py',
                    'paper3-round9/*/runtime/backend_entry.py',
                    'paper3-round9-thin/runtime/backend_entry.py',
                    'paper3-round9/runtime/backend_entry.py'):
        for path in Path(home).glob(pattern):add(path)
    return sorted(found)


def without_block(text,marker):
    begin='    # '+marker+'\n';end='    # END_'+marker+'\n'
    if begin not in text and end not in text:return text
    if text.count(begin)!=1 or text.count(end)!=1:
        raise ValueError('Incomplete or duplicate operational repair marker: '+marker)
    start=text.index(begin);finish=text.index(end,start)+len(end)
    return text[:start]+text[finish:]


def patched_backend(original,hook_path):
    text=original.decode('utf-8')
    text=without_block(without_block(text,NEW_MARKER),OLD_MARKER)
    # Exact baseline identity prevents rewriting another round or an unknown
    # backend version. Scientific files are never patched as a fallback.
    if sha(text.encode())!=BASE_BACKEND_SHA256:
        raise ValueError('Unrecognized Round9 operational backend; nothing was switched')
    anchor='    plan=json.loads(plan_path.read_text())\n'
    block=('    # '+NEW_MARKER+'\n'
           '    if entry.parent.name == "paper3r9":\n'
           '        runpy.run_path('+repr(str(hook_path))+', run_name="_paper3_r9_cumsum_fix")["install"]("paper3r9")\n'
           '    # END_'+NEW_MARKER+'\n')
    if text.count(anchor)!=1:
        raise ValueError('Round9 backend startup anchor differs; nothing was switched')
    fixed=text.replace(anchor,anchor+block).encode()
    compile(fixed,'backend_entry.py','exec')
    return fixed


def scientific_files(candidates):
    found={}
    for backend in candidates:
        package=backend.parent.parent
        for name in ('gp.py','worker.py','plan.py','entry.py'):
            path=package/'paper3r9'/name
            if path.is_file():found[path]=path.read_bytes()
    return found


def status(root):
    output=Path(root)/'runs/followup_20261025_r9'
    runtime=output/'runtime/three_jobs'
    progress=read(runtime/'progress.json')
    slots=[]
    for number in range(3):
        row=read(runtime/('slot-%d.json'%number))
        value=row.get('worker_progress') or {}
        slots.append({'slot':number,'job_id':row.get('job_id'),
            'status':row.get('status'),'task':row.get('task_key'),
            'worker_progress':{k:value[k] for k in ('status','stage','outer_update','session',
                'sessions_total','updated_utc') if k in value}})
    return {'complete_tasks':progress.get('complete_tasks'),
            'total_tasks':progress.get('total_tasks'),
            'updated_utc':progress.get('updated_utc'),'slots':slots}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    if args.status:
        emit('R9_CUMSUM_REPAIR_PROGRESS',**status(root));return 0
    emit('R9_CUMSUM_REPAIR_STARTED',imports_torch=False,downloads=False,full_data_rescan=False)
    hook=zlib.decompress(base64.b85decode(HOOK_B85))
    if sha(hook)!=HOOK_SHA256:raise ValueError('Embedded R9 correction SHA differs')
    compile(hook,'r9_runtime_hook.py','exec')
    output=root/'runs/followup_20261025_r9';plan_path=output/'plan.json'
    if not plan_path.is_file():
        raise ValueError('Existing Round9 plan not found; no operational files were switched')
    plan_before=plan_path.read_bytes();plan=json.loads(plan_before)
    folder=home/'paper3-runtime-fixes'/'r9-cumsum-v2'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hook_path=folder/'r9_runtime_hook.py'
        candidates=wrappers(home,root)
        if not candidates:
            raise ValueError('Existing Round9 backend not found; no files were switched')
        # Prevalidate every candidate before publishing the new hook or changing
        # any launch entry. No partial switch on an unknown old package.
        prepared=[]
        for path in candidates:
            original=path.read_bytes()
            prepared.append((path,original,patched_backend(original,hook_path)))
        science=scientific_files(candidates)
        if not hook_path.is_file() or hook_path.read_bytes()!=hook:atomic(hook_path,hook)
        rows=[]
        for path,original,fixed in prepared:
            backup=folder/'backups'/(sha(original)+'.py')
            if not backup.is_file():atomic(backup,original)
            if original!=fixed:atomic(path,fixed)
            row={'path':str(path),'before_sha256':sha(original),'after_sha256':sha(fixed),
                 'changed':original!=fixed}
            rows.append(row);emit('R9_CUMSUM_BACKEND_PATCHED',**row)
        if plan_path.read_bytes()!=plan_before:
            raise RuntimeError('Round9 plan changed concurrently; repair never writes it')
        for path,original in science.items():
            if path.read_bytes()!=original:
                raise RuntimeError('Native scientific source changed concurrently: '+str(path))
        receipt={'schema_version':'paper3-r9-cumsum-runtime-repair-v2',
            'applied_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'runtime_hook_sha256':HOOK_SHA256,'operational_files':rows,
            'plan_sha256':sha(plan_before),'recipe_sha256':plan.get('recipe_sha256'),
            'native_scientific_sources':[{'path':str(p),'sha256':sha(v)} for p,v in sorted(science.items())],
            'scientific_files_changed':0,'plan_files_changed':0,'checkpoint_files_changed':0,
            'result_files_changed':0,'other_rounds_changed':0,'jobs_submitted':0,'jobs_cancelled':0,
            'activation':'NEXT_R9_CHILD_ATTEMPT_AND_ALL_FUTURE_R9_WORKERS',
            'old_shared_hook_changed':False}
        write_json(folder/'APPLIED.json',receipt)
        write_json(output/'runtime/r9-cumsum-corrections.json',receipt)
        emit('R9_CUMSUM_REPAIR_APPLIED',runtime_hook_sha256=HOOK_SHA256,
             backends=len(rows),scientific_files_changed=0,checkpoint_files_changed=0,
             plan_files_changed=0,other_rounds_changed=0,jobs_cancelled=0,jobs_submitted=0,
             activation=receipt['activation'],record=str(folder/'APPLIED.json'))
        emit('R9_CUMSUM_REPAIR_PROGRESS',**status(root))
    return 0


if __name__=='__main__':raise SystemExit(main())
