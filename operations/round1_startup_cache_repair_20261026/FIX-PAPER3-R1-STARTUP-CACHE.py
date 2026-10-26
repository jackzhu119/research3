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

HOOK_B85='c-rk+ZFAc;68_F#fnz3fR4Xxd+Uv~eMxAp~C-Gd~#&$bxT@QyQA&F;-)BvPyHT8eLXBXf*7037T?eZa}$i-r@FAsJXa9#IQmRbrW!&EGjREo1VN6-KA9leOP2qTqiEpnBtk}OPTQMlQL$tuH#BFQtcRe2<}mfClm^K}BstyCd>0=p=T)>1_4Fk4A2vK)gVxh=vhk|LH-qCqwXRhmkrg)Wj5V{%NOq*JU(rlM4tj76M)K_S9bm}K~iZ;2A}F2Rs&C6+4xU1mbwB(Vf*UmRzJN@Oi{A)P!+cSN@lx{OHRWtrBC66;W}SqwcDw@I-UWoCqqAyilh;#(xePF%@l4%xyi7U8mxN*F1_*cUH(t;;Y;OC<#q5n?H6EH;S-*G%4kIuwP%3Rq@C=sT|KIxJWaEK4j`27yR6+gxF}S(X>9OYIn%D9;LcSER{RMPuJ=!Yo`#<=D4NETG8qR97!(g&0-+G=Ewz{;Bh<dd{_(Y#SEq)=VGqvm${G>=Wsye%{sd6>_sBXR3cE8|gS_FW(%$I}!`wZbP`qH#2oOBfpyw=}fOf_}z@1W_EMvI>DRcpFs>QkyC)gQ|`sV?62W$dHLzPZ|{Hm)txxO+4<2qs61B}K`d|Fsi3DM%k7g+ewX%fLt+G(wm+ll=frWGSS|%<S8x{?!K1}b=8nL>n=maUW}sI-t=F6I`(z21HX88Hf|Ik;<JTV3`ZQG@OFN$a!R&~!4`;_p<;rudKb!b4L7}nERhfp0gAl$8k9EyNO6oE#Ai>2-VzNS+X{O>@?j`~rNe?2)1nc6m5-!YkUa{LsFM%~C<dP(Ak`+9JUeXxDe7~q$MS&^}G}M7mX5kGyIJ`<_EhNRNWmoGnFI6N1ShlUuC*o{_U?ay$<pvzt$6M?En~fL3rO@KT41fXc)L`JZXSMF_8yE7(Rn3Gx1!r)zE2Q=&6Mrr5%xI5H4t{{PANU1_uMv+WhXW-=5O`Xq%c)R#ULc6WjnvyPg66-P3b{#&#rr&Ko0+J6B8IV82zn+Vbe`Tw>o+xF%@5S|bC-k`fv*kUXTnuwrk_<n5>$X305wTbX>fD+%q+4FrDsVN%R8Tf-0kwGMSws3>3y&eYUd8pJdFGD$BBX~EX1e#?t2~Hz)W!0%kKM4xZM)=jf{W`&|II~by8?|E)Ec(gq*Va!6pn8ZB;S9su^_ux#S9<WEt#ORzr~Ffh3n^j=lNe2t}|=OHIlqRPZ-Bf+f$AsK00?#kYjb)*J0K4#Wzh&n@>T)NbC98&O)D*+PqNAxnHOzWY{uB@PdsSAXV%HCy*@sgvi=qmsf3@{!4X5=DE>L>v>eYak^0;&|0yA6NrSh2}P)I}wY8u;@*jUze!%T|?LP?NK`s5A6Qf%PAM=&1qfFolFtxQ8Ah9U3WF^q8`(DoV+7c6`lpMsrZR7(>|XA7VYB^@HDw93-h}-*A@Wb43*tjS*A!K0D=}LS|B%S?<ma3r;u`3e5gRm;Dd%JQL#8zrJRh918EAhypchXdo?O_iJ#b9<SNgvuchRWi#V5nd<u@6U{*=kk607=ZC1m&-}i==yo(kfVf$#jiL$kf_LY{D=AGVaID$=Q<bBQI(Plg?Gm6_WBev(PEk~YaY*uA~#(E6x6n}AxOWSM)4r{x!0i98mVZ)_4M}?^-?TsHej7c~#Z7VgYXPET0WR>)o8J~#vug*l2CS=1+4lP1GHw11cKWMQDcZk8|l5$s52)zFIBhrW%`Mx+sG6JhhI9rPy*AbRqEwi9d2kMA?Nojh-X*-sI^jR^xKWlLts*RvHZn+`lHBdP+o@UYFYvKN3WL&uTL0^rUE)Fi6g{l0uvX%u0Mi)w%+UD|FFoYo%au}fS(}Ey?*+84k{~v&}W+0~{1Y{7>Fr}Q<NAZzv!Xl#l;P#>-yPS+NuDS6R*p}ns`DM@52mZs`RzYdnJ|Ak>@oZ!6O(es}e8}%QQ@;vREeBg4e;pZBrvw)!AGj1>FT~+z={orV)7Ls0ABVzq1>DtR;nWMAu9c=O?ei)KEQCy{hHle2V`AEzrnjI61*$86ZWL3zeW;39TTxOS!EtGd<hsVy*vA2pk#3^PRv`<^9^uohr|;5CF%?CLl-+o8Dl_|<9pt%Hfyw_{R_3Ckwb%2cHX?GxpL;=JNXAio$iI=->rK@Y8k4nU{u9ZJcXsiIm-J>|V2#f{2)F-HsC_QX_Jr7;@G_|Q$w1oNN<-DKaoV}4%wy4J-e{zoW43|y0-$<HZj|2+X}C!EEJYRR?ZmwoAWy^#Y)!yI8RaMpcOs>|h|1VBr|gCrEl7EyYt8+eyhN&ll96jx+TU&!%H5XdLp$UOjeryJ{etpsvP(bz7qdZ;0OD^B4(7wP@Mmx0jQdTUebli&nl@%^q`hSb$JA{{DqD}RSR+z(YnE)x2<EHjo*o9TPTn2|Kb*WiK06Oy9wF%uKAwJfeR_O$=1#dtUvS<>89uV5uNoUpngtp?^va*AVNz>-EXJDPlGHj16=eXOd~sMs*``;N73pwgtt8doTw`9(W~ZEIPfZ&!$gfAzZtev(Oz3Bz%jGh;^ITL_(X~7A^DP|7r4lLBg5C}HvnS$wt)xU?tJvgJOVsXgqaw$~jugljFG{_$?V!c??B<&-Y7&8DAD;&td8u$VOy|*mKA9!UJ!Bf@)7uX({}CMjEdFT!e*a>)#ciZoaj`yH=69+5avMXQcGwIkkr_rR*%p+Ow=GyT1YrVA&)!x^=yBotwpZa&>ZKNalMQEJqvYCO#;p4Ggzl{|1NIgfZswz<Cs)LA2?;(`)$QDtwqVedvh!XYi;q>2Ik)<_-`9j_u$kn*UI+IK0D+auX-Ul*h^!Z$JEd;qoRY*aP4n9z3$q1d`NY}J(`Q8U+}|>HNCxF7xlJLoSDRqk`0;SS-cTDNxgVu6R3k3(h(5}f)^sNv*JCAJd*A4HcY0gLlCqz4*mtSdJam)MEZQoz%+e&gZuB}-#6vl8-B+l=k`tCRwVOc^!nXjX%}Cnknkv=gJCxuV!tM-zuUr3ck#MWMYG7ATnwGv}m-I~88&rggW-4w(#fyG9+}qsP7*wGI{ZiTtH(YwiYUcI>a}W@y!@%G`U_p?%4sC8*h!@pNlvG$=y(X+&VuM`=VVZ=Rn!k;@`=5Hd6tz#P5+3B#2c{DkP7T78dS|<t*>NcCG(5x_SqmQqY2ZbR6<dRv+jWaYyd_BIvnb0(!<g6v5<b@=%|QWitC`aA1GDRH^0-X7g_C0Q^7Kyo=9TuTuTQ=1fy2|*>(Ow^G2QJqx4F7@g=I;>1W5>6J^Rgf#{tNWU@q)v<4lYXx?|ltI<rc$vI{fqW_WeXVb%nA3<4_4g8)#;9rv-dwG#gpXmdILg(BFKl&L-OskB>j+oRiIh+2;%4sEZk#*bqHmSN{@j$7w@HyGVh5zB4ooqT3g)t>i>GK;+~gTWFe`*~d<6t0`7ijb1aek_gV#FP%v`Dm(hN{q1$N?iw16Ipz|YN3%2i2PPojUqskr+i9cuM&0M6uXZy$Nm2U?7;wgRB-(Vab(tSt+=QR&ASUM&HVPbB!sL7+M~{4_w~UD8t#@4S4U?*zI%6c`ir}_7_y_y_Kp>01(7Zjna27{PKWrAZnT@sxr9ox!mHC-jl7oUK$>didLT*VQ?s&&K+hQ)63it%c<$3*W=}V>r}6pIH}j|O=1<Rl#TuDr#V6^o;FmD3mU`b$gsEB~Y#6D$sF$>-F-Y3*`daMgmdE^uu9pnYa~$XIy;~DxB)ZRXBRg4$XO4Z%O&31u9X<GUUH2nx+U6RGxi(`?u~kX8Qm9{1-&KJNCo!e|CZC9k68=g}^{vXUq))d&%!z}^g6eF%;~XTu!dN;ot*>(U<|#1#T!Tza+d`YGFuE?cpBrqzMLiD;X=`j7M#d^A(is&`h^(;_@O>WRQgK*m-&`92m-YpMdLe!gFmbS#m+591s!?8MRSl#2)=`g9eCt+#H#q%g51{`h-En@E$zQajZ7pnSnso5{N`jW8D19glF8e$`i#<m#&rg0jMp#neF5|$Ks0-?Dnc?=$K3!R**yfw*p9hqj2Ijtnxqj3xNdJxMZE(s;MY`;kM>Z?S5w)A-cXK0TQ|iKA1&I(>1>FY8<gK9IL7J@Ag}9>bN4rD9CvarKy?cOBi%C}|1G_)aX<D-FhseaBG;X3F-yi*Sbn^D-#apryuR5S>ISam<!@no+L~<{9;$whpAmWa|&@9Cnt$iP>9dnJxdUas$`gOo@91o8(Zc0}LQ8j5MxbmViXBp4HbVauVlQdQ`6Lf%^LMpWZA_6Z9-|V38%Ct!A)O>S}7KMsZ=?Y)ztRpX1Yxql-w4{s@MSDQ(OaR=jsV59q!W+TOWXiT<@B~VK+<{S6i{u-ZT&XZC`1&FND%P?MuadMrF0UZ_zlP9NfPF52=FYwo^?Ijivb;T5<nAfHJ2@bEe8q9pY3%$JkyEW6+bcxazpM5zw2D7F6afL?k}3$A%XYQrjtWpKpt?NCCN^O}w;6lB;g{rXmiCo(DaRhD6lQbdD4<J3jDGh`bDO)j-biPghF(<TE(ZrS?T?s&@ZRa??RN2ZeEZDg9L@hm7iMYLQ&-TrRGyK?e5;mXhni2^<SvjXBE2nki^GG?l|6$O{{sEjJMI'
HOOK_SHA256='12d26095b511cf608f4f5c7c6020fc38ca2402a60870eec254d4c1e3f8cf8440'
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
