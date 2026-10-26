#!/usr/bin/env python3
"""Attach persistent frozen-input digest caching to existing R1 startup only."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

HOOK_B85='c-rk+>u=mP692Bhf~O#WwzsR;Nw2`|VT(Gc^T6c=+b!C<20=^QU7fYmf~0J3bN=6NW=OrYc6^U-mk+U48V-l^_;F^)Fbq#*X(ZE9<ZPMclAXRidj6O1@I`8vNOWNgD|EKXl*ng^*lb0%Qt-iMg<@M>B+?jZ;vhI%XF$1?THq&OmxyF7S+W*tB@I&r3`*qI3YAEf$|N&DHV3Lam)bC6vmC}0FoBkVU8794)Jmo-&47VrVkI&Kf8krEnY_(lh+46wF8+{;$?GhYz&d8f%IZwkQd=1mD&Ha9hM6)!ftO`oFN&>&S(6xM%5E~dW~FjMry!KD4Efs3?%1VV79g8YDHBU8HFHvmG-j`8txJ*RrIrj7Az&$a$~Kt+u1a15wP04m3P@&07zbe(1|%5g%Mz9=IcM2sTWDCWQiUa5n!wQ{g|hP2=GkRM<KJwA5-X_#|8@xrutkxZ>IJS4MpZxEpVo_im_k+0g>jQ@g<ZF1`V2oS63~HvLjBawyLvt=H(PY3`ggXGL2&x&?eT{rwqW5_fSbIS>BAZN-3&=*W-Y+)X5=)p>%%bMZ;yWgV$c#g1&~BKJU^KIRm_$bU%vbH?)TrqNx)Cfj?RE83XSEdybh-fpE6bWCzJgl{o@A3;L7-)N%eCQ1VJj73}{#2E?|TR7eAQ?4E|k<yp%8lzKU_Z(S+V-OK@qYf%ufaJ3Tpm6A^8UQx&Q7<MAJu9b)X$>9N*@jzaIxE<RjP7+B}3%!Q^P1m6XZ4c$aqnliT_!TCzUWLCRrrtC`YCJa0hA6TXs>EfaiPN-dEcUu`{V2ug7B#N6UOGEG_j&aQQi@MbqsM0}09SEfo*Wkh8GMBZG7^|LLt<R#=iR56}wnCq<(+vb0I#w<>z>$2sweG+9ctKne9v@}^7@(aR4C3~z-o0bzLJ_*En=r=U<d-`uO*EOrYkBKNM`&{32hjE-y`b=Q;*sRwK$+z{GBRIInJx+oK`b`XY()Z^|7ObMCbNr=g=(7_sbeGtW04T}j6#?qzn0!_YQmZyP}k305?lm)ZTLQ8p)QqqRsjjG0GR`7qM~wsefZ2RvJRzZnM&ntj6oiD`QsvhKmO@`@DS?f=6NB~zWiyX!4($l%U$=q5!YZQaMz3O`;FLc5&K3)fD7PUU&3`}O*m%<5TS^ivid<L<QliClwQ>gy8l9S1)wAu{8&;0&k8QlrHNy2esBcKmw9PW*@z19rhs56RF?D?O{DNGBeV5J2aN-f!tis;J*?i%J90ym)@Jt55_`yE-?Q(&W#6#FgXh(s`C!f7{afm!`HQ5)umbte<S~k3qGlqB3EVXxB>Lh+)!-Pg1~3&gw-MolEf&n9H*S7iqS|*IUDvlq?SwpF_lLb4bAjHR-u1%C6hb|*lgZw7SL4p>F^$K`TSQeBc_5pzpAj?t^BKURe>?&_%`Qvpe)s0u0zd>qWj9uqIiwH(f*vPYAUA67L@4wrNI5({R3PQ>!GI`<U7YDsPR7RpX%1+4Ex9eC8Wp<4Pkb(NSrk`SQc}nzluLko8XPx+StVjW#2PDZR1NEK-y2%;E?R&D+ehO~n5`+?S6WiKcV@5Q5H^F6_cez{oAIPn7`J0aY|mL+4n57;tjYqN^$6NA{?Zngw%HsU)^;ZY2BRv24HxGe6{ecBH-6wSF5!e}Td7Gs!=$ezucXJ!h?xT{3UZ1Owz{~fR_p_eA{5h(_L8GkD8mpRwiM*>(*p^Bn}Y|RLq7ls%|P1H5ez^`#}xCz7!vq=BW!|u?9F*ab}<=cTyvvsq%Ft!^NXIl3~WHqZ!y)`XNQIz%{JyDNHX;9hiuoG`nAZ79Bd!-b!1e%t)9C*^@9C*!4B`G>+G>aU+ZXm929>mz%Dc9J3aU7T4~zSKCJ=+j>#PB&^BE#CZ^8|dJB3`pgN20MgrpHr>YS2RT<{u7V;}!Xx4S2fxQr5CNT|A`b>V|*`FU=H0$ZRG*L{MEg|i7o}5eNUz3A8_bPBXbj!+=xVQF8n$$-GfH8jV1%)FSN5UchM&775RZnP`tS$4OP-b}N7k_wxZ}tV&`0Rsl`yYkcePOmI#P)=jL&Yx+(&k<otSFt+&RJz1k2dp0Bi%f$4Xoz?su$=+#m$h06772_sz`4qzr6r?!d@07Vxde5D6V!a$Hoqpf_Z@%0yZlkZAh*R^+1Xe(gi5HsP6mt`jp*hC>dLx586RjXat<F?-!V}qFwqqG@A_)Z6JPea4;XPg+51<VBBx&9lVb9(bP+kk@lwY6jL`HseDdPVvR`E&HL4u5!}}(JUQgA-@QNPKfZf&e0s)T9YNyEKc9Shb8>uo8cwOuT~O|XSu$iC-!wLyGz)Zk=#>yv)1lV-Sd2BnC9(Y?H16@8oNHK#_{LG9Y(89BE76NLS1@nnGd#*wr>+&pi>r~;mHKH76a2|dxm;$qQ3#buaurVEVhfHGVwoWf!*|2|><K$tYbhbHb!yl~>_Ceu12?h-Z0u{PWA?H%JKs`Sd{1t^*+O2=AYq2j1CF#*a5k7O!vA74iz*_>G~B26pI-f)AOFJs+5Y|U<#3DJNWG$hceGUQQpM#q1$o+GGoVbRNp!ZgnAf%~ST)ta1O}g@t(Kt2MHu^fKg85aFZw1M&Ok;fOuS4<_3a768)pXCTcp^`N6R;=D$^1q_)=Afb6*|;gQgsu_i91-SQTn>uaCQZg?xj}C<p9y;GP^HuyQ#gvEczCn?)2(v7<M~BvIsfal@5R3&QeAu%Bm4h!%yvt8hpL%F%M0gV0fJf@$N&!vT9kZHVM?l1rgST;vhGh_9`S-cekSl|}7+qu(LvZ5>O@e)3`OqF(dRO-8e5uh>%MnYwE9I#k3%IdNT^*I-ErOP>49zzXm!0Mlk9?sHA0=kgs)@El=xh9BzoIk<?}ny4Ds6%?n%@32cou5{!T;i8#}nmOsBUsCoqH!=oRD7;^ey5WvX4_V#ZF?Y>$t`E7x0q#MNxDIXZ+i>UAOqf)7UcE-9T*3ys<|5C8!3Juh?*6CVE`=?Kstg8l?4{x<2Biif$Ntl9=5`!RI~@<UhOC7SR1DxnixpV|8^m>+g1tvbr}G@o#(*(l6UgXn2sH-^NLvkwjvu&PcT=QgPHm7JHm}I+Ozd8n7<=Q`FCI8NZoLr>#~d@<zPKs$RmeO`3?`5Su{HAF;&2>*{0QPgjyBH3`CvHK0mE~jOqE@jX*ZOsqXM@kfJe@;Eax1clDgJoYilL`EzqWN{wqcBrx8<sYEo&p=C((-!w~fzi5%KqTa6#b1XzZjw>h$$@7-WbPemlRpLcT4sH(*sBc)2CE`z}mCi{6^LMU7{QI$YSF8i@GniFF>glCMoDKIgHZBUvzkebM%GfEGQbjT97vT7&-40_6!ER8Br=S{KuD0AHXKfoRgutx>ge-KA*{oaa;%FwjCfTfAw9+w0m>w*5baM;0oFoK4=<-^s{=}#X%9G(0c?k$G=Xt%w?ij+mtWhV2~e9Z|99dwO$WGR=>nX*xx)*8rbX%0wJ-CPeOsdVyG+63q+V}k^9i4Rfu^w-(b&FpD<_Vn%i>4*8#)8Al?M6;sPXt3bdFt3(+*H478T0v|Wsl2e4w5KtM+KBpE?B|xp^arnK49`S@;D_GL12hue3AvV?EF^MAr|z~09hZ(Co`zxg88>Zr6~kRUaVNpL#G4)1uc+^PfD30Srv5IUNNfrI`sVXb*b=yuG2W<fU3-@W)!BH*IZ%8JWAWIrz77$)C+_^Y2AP_+g*KNWxhl8!4R+w7pL0jr8rz1Eu?h-xhQ$*^*4QcXK932pIIOhqj^2Sw`w+ffNX$7A^S!)WH_K6t@^Y(c7~QpwOPu0cC+pGR1f4tp|C@Bj#Z@MM-jcSpu&Ha(f!|jWv?PVn2gBf^&+}gFIeK;W?&o6&ODx=F%6*Bt!0whC?%(Xwl~;;yzPbK6$K;f|I}7d#PrD%fH>$V6NhKEPvRfYcte}9X-DH2bTNRtqSbv=&fw;1Gqe96WhP{J4Tdgg-#O_DCLqexmXu`dFfK!W0S0)3yKhSAflI@4cM4&WoqMtq<{e1N9{n5+!XeU}V$Ez(1y5B;-C*X<bUf_w30kVOJI|M_wlwh>>eXMreRT%Hp+~1e$fa5qG9xG}}R|QcusbW-l;km6$XTWrYH|(-J)lxA$1kFJzy#Yi7ytJ{~K|hqa&HU7K`wSNa6{R*6zT#O&QLfhDFJ00SGfF7hIkIyBaI?mqFt`%DVbn~<Y&!*>fYP6Lz)001#m*&HS}03b01;4OEn9J!<@J$s1=;^KgsuYYz5tpB`%b0nou<k1_Fz%Cr}*yVfaviR$5E&8^H)ScwR+fIL4^IgY7axJ`13;{AON_e=DfMER(tNK0JQ?D3y5T57Y2AEu;&|biQeXEUwfAd{DF#fn;S&|Uc@2vyK9=;)V+;HI@>t(yc%~gIH+lVgc%U;gMQxb7k^7PxLnTB{O@$(mIiz33Un@Y<m7SRs-^g$?h`e+EhLJN-rC*b@St<;&Edtr0Oid7xB'
HOOK_SHA256='fb704ad3030f27142ffc4c9694b3f4f46cd4cc065eb33c4341721ac6424a646b'
BASE_STARTUP_SHA256='e2141c386bf7590e4d738e11661b7f0cd297c06b9f4ae5ed7fa5c2c72fe1df42'
BASE_BOOTSTRAP_SHA256='43513735525e678138d23fa848d901c3c780ece11a57cba3d675f158d13ca9d8'
MARKER='PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1'
SPAWN_MARKER='PAPER3_R1_OWNED_DIGEST_CACHE_V1'


def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields),ensure_ascii=False),flush=True)


def sha(data):return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary=path.with_name('.'+path.name+'.r1-cache-'+str(os.getpid()))
    try:
        with temporary.open('wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,mode);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def write_json(path,value):
    atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def option(command,flag):
    try:return command[command.index(flag)+1]
    except (ValueError,IndexError):return None


def packages(runtime,home):
    """Follow only R1 launch and durable scheduler package declarations."""
    launch=read(runtime/'launch.json');state=read(runtime/'scheduler-state.json')
    values=[launch.get('package')]
    if isinstance(launch.get('worker_command'),list):
        values.append(option(launch['worker_command'],'--package'))
    for slot in state.get('slots',{}).values():
        for attempt in slot.get('attempts',[]):
            if isinstance(attempt.get('command'),list):
                values.append(option(attempt['command'],'--package'))
    found=set()
    def add(value):
        if not isinstance(value,str):return
        path=Path(value).expanduser().resolve()
        if (path/'round1/action_startup.py').is_file():path=path/'round1'
        if ((path/'action_startup.py').is_file() and (path/'kernel_bridge.py').is_file()
                and (path/'worker.py').is_file()):found.add(path)
    for value in values:add(value)
    if not found:
        for pattern in ('paper3-r1-repairs/*/active-round1/*/action_startup.py',
                        'paper3-parallel-migrations/*/round1/action_startup.py'):
            for path in Path(home).glob(pattern):add(str(path.parent))
    return sorted(found)


def patched_startup(original,helper_path):
    text=original.decode('utf-8')
    begin='\n# '+MARKER+'\n';end='# END_'+MARKER+'\n'
    if begin in text or end in text:
        if text.count(begin)!=1 or text.count(end)!=1:
            raise ValueError('Incomplete or duplicate R1 startup cache marker')
        start=text.index(begin);finish=text.index(end,start)+len(end)
        text=text[:start]+text[finish:]
    if sha(text.encode())!=BASE_STARTUP_SHA256:
        raise ValueError('Unrecognized R1 operational action_startup.py; nothing was switched')
    block=(begin+
        '_paper3_uncached_startup_active = active\n'
        '@contextmanager\n'
        'def active(root, action, *, modules=None, torch_module=None, emit=None,\n'
        '           action_file=None, traceback_interval=300):\n'
        '    _cache = __import__("runpy").run_path('+repr(str(helper_path))+', run_name="_paper3_r1_digest_cache")\n'
        '    with _cache["active_digest_cache"](root, action, emit=emit):\n'
        '        with _paper3_uncached_startup_active(root, action, modules=modules,\n'
        '                torch_module=torch_module, emit=emit, action_file=action_file,\n'
        '                traceback_interval=traceback_interval):\n'
        '            yield\n'+end)
    fixed=(text+block).encode();compile(fixed,'action_startup.py','exec')
    return fixed


def bootstrap_files(runtime,found):
    launch=read(runtime/'launch.json');paths=set()
    for package in found:
        path=package/'r1_spawn_bootstrap.py'
        if path.is_file():paths.add(path.resolve())
    if isinstance(launch.get('direct_package'),str):
        path=Path(launch['direct_package']).expanduser()/'r1_spawn_bootstrap.py'
        if path.is_file():paths.add(path.resolve())
    return sorted(paths)


def patched_bootstrap(original,helper_path):
    text=original.decode('utf-8');begin='\n# '+SPAWN_MARKER+'\n';end='# END_'+SPAWN_MARKER+'\n'
    if begin in text or end in text:
        if text.count(begin)!=1 or text.count(end)!=1:
            raise ValueError('Incomplete or duplicate owned-stage digest cache marker')
        start=text.index(begin);finish=text.index(end,start)+len(end)
        text=text[:start]+text[finish:]
    if sha(text.encode())!=BASE_BOOTSTRAP_SHA256:
        raise ValueError('Unrecognized R1 operational spawn bootstrap; nothing was switched')
    block=(begin+
        '_paper3_original_bootstrap_owned_stage = bootstrap_owned_stage\n'
        'def bootstrap_owned_stage(root):\n'
        '    result = _paper3_original_bootstrap_owned_stage(root)\n'
        '    if isinstance(result, dict) and result.get("status") == "NOT_APPLICABLE_NO_PARALLEL_LAUNCH":\n'
        '        return result\n'
        '    try:\n'
        '        __import__("runpy").run_path('+repr(str(helper_path))+', run_name="_paper3_r1_owned_digest_cache")["install_process_cache"](root)\n'
        '    except Exception as error:\n'
        '        print(json.dumps({"event": "R1_OWNED_DIGEST_CACHE_UNAVAILABLE", "error_type": type(error).__name__}), flush=True)\n'
        '    return result\n'+end)
    fixed=(text+block).encode();compile(fixed,'r1_spawn_bootstrap.py','exec')
    return fixed


def preserved_sources(root,found):
    values={}
    for package in found:
        for name in ('worker.py','kernel_bridge.py','audited_runtime.py',
                     'hardware_comparison.py','scheduler.py','START.py'):
            path=package/name
            if path.is_file():values[path]=path.read_bytes()
    for name in ('p3work/contracts.py','tools/r2_backup.py'):
        path=root/name
        if path.is_file():values[path]=path.read_bytes()
    return values


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    runtime=root/'runs/training_campaign_parallel_v1'
    emit('R1_STARTUP_CACHE_REPAIR_STARTED',imports_torch=False,full_data_rescan=False)
    helper=zlib.decompress(base64.b85decode(HOOK_B85))
    if sha(helper)!=HOOK_SHA256:raise ValueError('Embedded R1 digest cache SHA differs')
    compile(helper,'r1_digest_cache.py','exec')
    folder=home/'paper3-runtime-fixes'/'r1-digest-cache-v1'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        helper_path=folder/'r1_digest_cache.py'
        found=packages(runtime,home)
        if not found:raise ValueError('Existing R1 startup package not found; no files were switched')
        prepared=[]
        for package in found:
            path=package/'action_startup.py';original=path.read_bytes()
            prepared.append((path,original,patched_startup(original,helper_path)))
        owned_bootstraps=bootstrap_files(runtime,found)
        for path in owned_bootstraps:
            original=path.read_bytes();prepared.append((path,original,patched_bootstrap(original,helper_path)))
        science=preserved_sources(root,found)
        metadata={p:p.read_bytes() for p in (runtime/'launch.json',runtime/'scheduler-state.json') if p.is_file()}
        if not helper_path.is_file() or helper_path.read_bytes()!=helper:atomic(helper_path,helper)
        changed=[]
        for path,original,fixed in prepared:
            backup=folder/'backups'/(sha(original)+'.py')
            if not backup.is_file():atomic(backup,original)
            if original!=fixed:atomic(path,fixed)
            row={'path':str(path),'before_sha256':sha(original),'after_sha256':sha(fixed),'changed':original!=fixed}
            changed.append(row);emit('R1_STARTUP_CACHE_ENTRY_PATCHED',**row)
        for path,value in science.items():
            if path.read_bytes()!=value:raise RuntimeError('Native R1 source changed concurrently: '+str(path))
        # Scheduler state is live; record its original signature rather than
        # treating concurrent job progress as an installer failure.
        receipt={'schema_version':'paper3-r1-persistent-digest-startup-repair-v1',
            'applied_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'helper_sha256':HOOK_SHA256,'operational_files':changed,
            'owned_stage_bootstrap_files':len(owned_bootstraps),
            'preserved_native_sources':[{'path':str(p),'sha256':sha(v)} for p,v in sorted(science.items())],
            'original_metadata_sha256':{str(p):sha(v) for p,v in metadata.items()},
            'scientific_files_changed':0,'checkpoint_files_changed':0,'plan_files_changed':0,
            'jobs_submitted':0,'jobs_cancelled':0,'signals_sent':0,'other_rounds_changed':0,
            'activation':'NEXT_R1_ACTION_CHILD_ONLY_CURRENT_CHILD_PRESERVED',
            'cold_cache_behavior':'Successful native hashes are cached; the first uncached read still uses the original hash function'}
        write_json(folder/'APPLIED.json',receipt)
        write_json(runtime/'digest-cache-repair.json',receipt)
        emit('R1_STARTUP_CACHE_REPAIR_APPLIED',helper_sha256=HOOK_SHA256,
             packages=len(found),owned_stage_bootstrap_files=len(owned_bootstraps),
             scientific_files_changed=0,checkpoint_files_changed=0,
             jobs_cancelled=0,jobs_submitted=0,signals_sent=0,
             activation=receipt['activation'],record=str(folder/'APPLIED.json'))
    return 0


if __name__=='__main__':raise SystemExit(main())
