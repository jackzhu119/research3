#!/usr/bin/env python3
"""Stop R1 automatic child refresh; replace only its owned CPU controller."""
from __future__ import annotations
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

HOTFIX_ID='paper3-r1-refresh-disabled-v1'
OLD_PACKAGE='2bfd664251ee5d1bcaf5'
OLD_MANIFEST_SHA256='f30f3e60102b7bfaa54d758a69427f7e9fc27d7330ab22c59d9a458f08c9fbc4'
OLD_SUPERVISOR_SHA256='b09934e9f6246492388ba57159e1f2af41749c5a857dc9410f3252d3995bbfc9'
SUPERVISOR_B85='A|fJ0WprU<Xk}y|QZXQMWn*u4Wpa5sAaZ4Ob7df5Y;131VRUJ4ZgVUka%E<6WpijCZ+C8GWFTQ<bZKvHAY*80Y-Dm}ZZ0AsA_``5Z*3r7UuJc5b#i52Um$61aBp&SAYpE9Z**aFX>V?G3TbU{Z*p`XVRC11VRCb23TbU{Z*p`XV{dL_b#iiLZgehYb#!%dWpfH?ZE$aLbRcG9Zggx4X>D+Ca&#bQVRL9~X<`a#ZE$aLbRcSTZ*B@{ZE$aLbRchY3TAR|Z6I)AbZBg8VjyX4aBp&SAW&g+XbNd<aBp&SAaiMFZeeT+X>D+Ca&#bbbz*RGZ)0V1a|&r~aBp&SAai+h3TbU{Z*p`XbZByAVPt7;X9{U;aBp&SAarSMWeN&NMN&mQP+@dvC|_S@X>4U*UnwqfWpi(Ac4a6jE^uLTWo~o|b9r+vaA9<4E@^IaWpZ>VFf4O)awteeQbj2#3TbU{Z*p`XaA9&`Y;0w0UvhPBbZKp6AYpSLaB>Q1ZE$aLbRcPNb97;BY+q<^Z)<Z33JPRpW*})~Wo~q7ba^OnX=E&La&Kc_a&K>RJy2nEXecKyaB^>BCn+g93LqdLAarthItm~lARr(hARusQWIbtabSQ9XWGM<DARr(hARr)VW*~5BWIR1FI&x)nb#iVXPH%2y3LqdLARr(hAaFfUVRUFHaB^>BUvh76bSW=$baE(gX=Eu1ARr(hARr(hX=WgBE^~BYbSNn<b97&IX=EWiZ*wkZWps6EWGE>*a%FUNa&90_Z*FA@ARr(hARr(hbY*ySJt%N5Cv$XRbSEh;a%Ev;Uvy=7bSNo1AZBT0Y-DpibY*ySTXbc3bS`pcX>MdFCn+Z>D>6D=E^}~fX>=$l3LqdLARr(hAZcbGW@%+?WOG|ET|GS~S|>VkWps6NZXix?Ze<D}ARr(hARr)Na%XluTVXC_Wn*t-Whf~iW^ZyJVIXO4ASiGzCu417Y-w&~Cn+v+WnpArVtI6Bb0{e;b8u{FbSPpcTrejoAZcbGVO<I!ARr(hARr)eWps6NZXjf7V{|BRX=FWcX=E&QX=FWcE^~BYbSNn<b97&IX=E&ObYXIIUvz0>YjZtjX=Q9=b6YVvT`Xa8XLdbda%Xla3LqdLAZ2)CWpH#LC{I&Ga&m8SELLG`b!A0za&K}hNp56ictvt@Z*nYEZfRq0WMxHia&K}eI&x)nb#iVXPH%2y3JMBjWo966aCB*JZYW`LXLc-WWqBz&3LqdLAZcbGVRC17E@N+XZgePXWqBzfJuy0RWps6NZXix?Ze<D}ARr)VJz;WZb}ngdWMz0LYh`&U3LqdLAaZ4Nb#iVXVRC17TWKpXT_9;@AZaTxJZxodC}DDEb}1lbY;$EGPH%2y3JMBjWo963Z)0m;Y-x67C~#qPXel}hARr(haA9<4Jy2nEXee-DbZ99GARr(hX=Wg9Z*(AVVRUFNWq4_GbaN;vI&x)nb#iVXMqzAoWeOl5ARu>XbZ8)OVRUFNZ*XO9C?|3!DIj5UAZTH3WNc+R3LqdLARr(hAarthI%Z>TbZjnWY;R+0C}?4BWNc+DW@B!2Y%WYsLrY&pSbSz<ZggxeOix2gUrs_P3LqdLARr(hAZ2)CWpH#LLTqniYiVw0Nl!&`a&K}va%FUNa&910a&=`2ARr(hARr(hW@B!2Y%XSOZ)0mHXkl(-Y-KEFV{UY8E=*5DOJ7w^DGDGUARuyObairWAVy(qb7cw&3S?zwAa8eWWn^DtZ*FvQZ)|L3awu?VWGr%TZ*(kga%pF2ZeeUJDlBkvZ)0C_Z*O!xP+@dvC?_v)a&KcNDJeP%ARr(hc42IFWj$$RWo~q7ba^OnX=E&La&Kc_a&K>RDGDGUARuXGAa-GFb!8xFb0AJ{Ze==hWps6NZXix?Ze<D}ARr)NJ$7Mib!A&8VRC17CtW)ra%FLKWpi|8WIb<ibZKvHC}AuoEiH0yZ*(Uq3LqdLAZcbGZf|rTa%FLKWpi|8WFT*HAW&g+Xee@Jadl;LbY)~IE^=jaZ)|pDC@CR5P+@dvC~|LabSW-!Wpi(Ac4a6jI&x)nb#iVXPH%2y3LqdLAZcbGZf|rTVIXgEAW&g+XeeP@FkLAwa%FRGY<6WRDIq;jVRUFHb9r+vWq4&{b#!53Y-K4fa%FRGY<6WRDLQgxbairWAWm;?WeOl5ARu9MJuwO(ARr)jXlZO^AYpVoY-MgJVJRSCZe$=~TVZrvAZczOC?_pRCoCr|QztAZEkY+OCoOd+EGI2RCoCr|b0;h(El(#bCoNA;Cn-8%bSphE3LqdLAZcbGVRSw{Y-MgJVJSLtWps6NZXix?Ze<D}ARr)fV{&P5bUjdEbZ97HTVZrvDK2tlb8l>RWhf~MARr(hZ){{eb7OL8aCALAP+@dvC~tCUXK8L>Y$+~sWpi(Ac4a6jFDFw}K~hvMaCs*nVQyp~CoL^wZ*FvQZ)|L3awi~ZZXjV>VRS1oI$a=PZe$=QEiGecX>4RCAZ~ATAZczOVG1B1ARumKcRh1sa%pgMJv~T8QbjK(b9HcKa&~ERZ*nehc_$!YZe$>AWo{^8DL!FzD={EpZe$=~TVZr7F<m`9Cv9PFVP|D03LqdLAaZ4Nb#iVXWNBk`D0X3Nb!9ALb#iiLZggL8VRU0?J#J-pDIjTPAa87BAa8OYZe@2MWo&b0AWm;?WeN%kWMyU`W@&C@Ut@1>baHQOY-MtDC~|LabS!UjX=iC}VQegLa&Kc_a&K>RJy2nEXecKyaB^>BCn+g93LqdLAZBlMZe%@MT?!x|ARuOMav*RZX>K4;VRUFHaB^>BUvh76bSW-rbY*g6X>uqjItm~lARr(hARuXGAaE{jVQpnDX>(+0XK8dODLM)uARr(hARr(hARr)iVQh6}J#TkzWn^DtZ*FvQZ)|L3awu>vZeeX@EOKvebS!UjX=iC}VQegLa&Kc_a&K>RJ#cbwV_$M_Z*(aNARr(hARr(hARr(hX=WgHVQh6}I%aQmZe%WDaByXAWGHrFY;|QR3LqdLAaZ4Nb#iVXW^Z+FWC{ujWMyU`b98TTUvF$=Ut@1>baHQOY-MsNa&LDma&K>REN^mYXK8L>Y%D4)b7gL1J#TX^YiVq3EOTsSWpF)oX>DaLb8Ka0a4chNZ)0mcbZKp6E^TjaZ**^NX=5om3LqdLAY^4`Aa!nIXkl(=Wn?HRItm~lARr(hARu;OY;|QlZ+C8GWM5-%Zgg^QY;0w6C~|LiTPJX7WG7uLa&K>REN^mYXK8L>Y$*yLARr(hARr)eWps6NZXkAHY;|QIX>%ZMZ*(9|Z*FBEVQyp~VQg$Dc42IFWm{`qJw0-7cUx;+AZBlJAZs9LZXhTpaA{;GEGKnoWG5^qb97;HbYFC7V{3CKEGJ=dXLctkDGDGUARuXGAZ~ATAa!nIXkl(=Wn?HRI&x)nb#iVXWNBk`D06gSbait*Cqqw8R8mh&Ohr;(Nkm0XR7q4>UqeViPDe#VCoFSmXKrC^b3I#KDGDGUARu&dc{+1tZe%EOZ+BZKaA{;GT`Y5HXKrC^E>lTIR7FxvDGDGUARuLUV`Xr3AW(8|V`X!5OmA;%b#O&;a&K}va%FUNa&91GX=8LKb97;Jb#px@Lr+drQcp}wMN(fuOj1QbL|I=&PDDjSCoFSmXKrC^b3I#KDGDGUARu#TXKrC^b3I!pQ%OfuMN&;CT|0GdbZKloV{C6@YbYrzF)#`sARr)jXlZO^Aa!nIXkl(=Wn?HRAYpD~AY*KAV{0fWJauk#X>2-kY-MF|C@wK63LqdLAZcbGb#7y5VQyz-WGE>*3LqdLARr(hAarthI&)=iWGHfPcUvcLX=Eo|EOTjRZeeULQ%OflNlZ*BJ9BAgZeeV5E@5zRWo~3BCsRpBOG!*jCn*XbARr(hARr)Rcw=R7bRbZ2Z)0V1b4+h<Yjto%a&m8SI&figa|$3JARuyObairWAY^G{bSQIlVRUtKJtt3BPDMmtO+ijUM@3CVPE=n*Pfk=)PfScjQeRV4Pf%Y{MNw5nQ&dGnCoFJjWIb|kcUvcLX=Eo|EOTjRZeeV5J#%SiZeeV5ED9hXARr(hARr(hARr(hARu#NX=QG7X=Z6-Ut?%#Y-Dm}ZeMe0XKrC^Y-MCUFf4Oyb#iTAVQg$~V_|e@Z*Fs6V_|M%Wo&F^WIZq`3JMBjWo966a%pF2ZeeU+aA9L>VP|D1a&K>RENE|SWhpueARr(haA9<4b3JKpb97;BY+q<^Z)<ZdWNC9_Z+2yJUvOb#Yhh<)b0~6eZ*(kZZ*65M3LqdLAZ%fEZewUYaB?nkWnpArYIARHC{ST^Xee@TZ*(ayaB?nBP(@NfR7p=xK};_vY+-e7V`wgFb8l`ZEPH({a%FLKX>w&`JyddaWhn|EARr)ca%E;^a&l#4J#1lhZewUJXJvFKCvah7Yhh<)Cn+Frav*PTbZKvHC~RSMZewUJXJvFKCwFghYh`j@V{dJ3VQyq6EL&YEEGI23aA9L>VP|D0DGDGUARuOMav*RZX>K5JVRUG7Itm~lARr(hARuXGAaHVJW@U16Wn>^>Ze$>EE^=jaZ)|pDC@DQXP+@dvC~$IRW@U16Wn?KXa%FRGY<6WRDLQgxbairWAaDvGARr)VW*}^3ZYXeJbZB!aJv}ixa%FUNa&91SVRUG7TQFS;ARr(ha$#w6Wgu2zY;|Qta&m8SC?`c&K|@quLse2zMNU*-QZZjoP(@NfR7p=xK}=szK|@PHM@3&!MNw5rQbj~3DGCY-WMyU`a%E;^a%FB~WnXf2ZggpFWhioQZ*(a-3LqdLAaG%HXgyG2bZ97YZ*O!dFK}`$Pf$ftK~zakPC-mBCvtUebZKp6UvzR|ZgXa3a$jR_ZE#_7X>)IGE^2dcZYK&LARr)WZ*_8RVQf8caxQXZVPs!wb8l`aaA9<4EPH(^Aa8OYdwmKZARr)eWn*u0WMyPMYHxLNZeeULXJvFKCum`EWOre5WnW`#VRLg|aB^>EX>4UDDGDGUARuXGAZc@HZgX^DZewLAa%E$0a%5#>EM#e8bSXLtARr(hARr(hW^!+BAZTH7WOre5WnW`&ZE#_7X>)IGAZcxIZ*p`XUvP47W@&6?EFfQUVRv6}a&Km7Y-I`{ARr(hARr)SZ*m}ZVQh6}AZczOYHxLNZeeULXJvFKCvReNWpZ|5bZKvHb0;i&eJL(>VQh6}b0{e~3LqdLARr(hARr(hAaHVTW@&6?J$7Mib!9GRWppSfa&>NWX>Da+aB^>EX>4UDEPH(^3LqdLARr(hARr(hAarthIv{OfbYo~`b3I>ha&Km7Y-K2La&Km7Y-K4uJ#uAZZ*pX1WFT*HAYXD}cVBRFZ)Rz1WhiiRZ)Rz1Whp&9a%E$0a%5#>3LqdLARr(hARr(hAZ2)CWpH#LC{|%?b!A0za&K}hOJ#XQa&m8SEL3@LWkqswZ*nO*ZDDj{Xk~LfMqzAoWeOl5ARr(hARr(hARuXGAZ=lEV`yb_AYpD~AaHVTW@&6?E@x$QC?|AjZE0?2UvP47bZ=vCY$quoVQyp~aB^>EX>4UKXJvFKCuC)IX=7zyb#-ZECn-7#ARr(hARr(hARr(hARr(ha%FUNa&91Sa&Km7Y-KEDX=8LKaA9<4J#%z&C~#qPXelgnZ*_8GWj!ZlcxiKVX>Mm<aB^vFVRCt2a&>NWX>Da+Z(?(0a&}>KX>V>PDGDGUARuOMav*RZX>K5MZ*p{HWGE<5VRUFHa&K>RDK95-b#8MnbaG*7ZfS03Ut?iyaA9d@ZeMmWFK}{iW@&6?b0;Y-XKZg`C?_f}a%E#>X>fEdYIARHCn+g93LqdLARr(hAa-GFb!9zpaxQXZVPs!wb8l`aa4dU$DIjlhAbWi~aB^>EX>4UZc42IFWiDrBbSNisb#8QNZDn6@a&Km7Y-J}Y3LqdLARr(hAZcbGc42IFWiDrBbSNitbYXOLb0;Z1Jtss(Rz*xtP)$WnR9{0uOi4mgK~zakPG3+`Pew^hMPEZtO;AilR7EEsVQyp~X>)0Ab97;DV`V6Ca&Km7Y-KEDX=8LLItm~lARr(hARr(hARuyObairWAaHVTW@&6?EM#e8bSQ9PbZ9+ubaE(gDJ*kub#h~6Jtt#tZE$R5bY)~;aB^vFVRCt2aB^>EX>4UDDGDGUARuyObairWAWm;?Wh`WAV{|BUZ*_8GWj!ZQQb|ogQdwV8RZdh%O+{Z$PgGx7MO0rwRzXQjK|)MLCn*XF3S?zwAY*lMa%FCGUuth+b0~6kZggpFWhpueARr(hb97;JWj%0mE^=jIWM67?Z*C}1VRUFHa&>NWX>DaGFDG+jXk}z|Y-MsSb97;JWiD!SZ*C_ndwnc&WpQ<Ba%E&aRC0A?DGDGUARuyOb9HQVJzHH0ARr(hW^ZyJb8K&PAZczOa$#;~WhgT#Itm~lARr(hARu9MbY*RDbaOp(bYXO5E@x$QC?|7lZ*+4fEPH(^E@x$QD06gjD06IYbSW%*eJL(yWppSfVRUq5ZE$pRCoEfCDGDGUARr(hARuXGAZ~ATAYpWLWo>YDb2?*hZggpGb!7@5ARr(hARr)NJz;coWo>YDb6YJjT?!x|ARr(hARuXGAYm?NWppSfYHwm+X=EoUAYpD~AYm?NWppSfb8K~TZC`VAVRU6DDLp+WQdLe)Nlr&6AYpD~AYm?NWppSfb97;Jb#o^vAZ~ATAZczOC?`fqPDxWpMMNhoCsIX9MMG3YL?<aa3LqdLARr(hARr(hAaZ4Mb!>DlVQ_F|Ze%DZb8K&PEOT^nC}CSCYHwm+X=Eo|DJdxmARr(ha%FUNa&91UWpj0GbP5UzWMyU`Zf|5|Ut?ixY$$DSWMwRJZ*O!ga&>NWX>DaJb8K&PENX9JEO2>rXm4&TDlBq!Zf<3AJ#%$paB^>BWpi^ba&>MgItm~lARuFJZEaz0WIbCab8>ZVCoCr|EoyIKX=FVoD{60IEGI23Z+2yJY+-OGEGI23Zf|5|b3HL9EGI23ZggRDYjZs@CoCr|En{$Xb1iUXaxHXWb89^@CoBpeARr(hARr(oEiG+jZ9OL|C?_&UCm?BNAZ>4CWj#G7aB^>AWhWqIY;$EGCo(lQO(!WVCoL^!a%FQpCo3o?XK-~oF()8tW*}{EWMw@)Cvb9aVr3^FWo&b0ASZ5bZe=GaED9hXARr(hARs3#Ep%yZWj!!3IxsXkFfb=9aCvlSZ*D9nElDRVCoNMaEGI2OCoFSxawteeQbjK(Ze@3FZ)9a(bZ>8LE^v7#DJ*SoWMwQ0ARr(hARr(hCoL^<Z*O!bEOT^nC~|LabSW$+EiH0&ZggpFWhX3ibaE(ib#8QNZDlDeCoL^%Z(=QJWG5_YZ(=MbEiH3wZ*(Utb98bjb8K&PDP0O6ARr)ga(OxmARr(hARr(ha%FRHZ*FsCJ#uw!Ze?;PV{dJ3VQyqBV_|S~b#i52Z*_EVb#y&ca&=`abY*ySJyddaWh``QZDntDbUiURFd%7WAZ>4CWj#G7aB^>AWhWqIY;$EGG%zU&ARr(hARr(hW^ZyJY-w&~AZczOa%Fa9a&u*5C~{?UaBpsNWiE4cWN&qJE^}~fX>@F9Ze?>QDJeP%ARr(hARr(hARr(hbaHt*c42IFWj$(hZ*DGZZ((F}C~RqNWhn|EARr(hARr(hARr)Rcw=R7bRbq?Y;|Qta&m8SI%98cbZKsNWeOl5ARr(hARr(hARuXGAZc@HZgX^DZewLAc42IFWh`WAV{|DXVQyp~b98bjc42IFWiDrBbSNikZ(?6*WG5*pJw0k~Vjy8|WFU57Y;|QWXJvFKCv$9XbSEi2J#%btbUF$kARr(hARr(hARr(hARr)eWps6NZXjf7V{|BXVQh6}EOT^aa9?s|bairWV{c?-J#uApaBpsNWiE1MbairWV{c?-DGDGUARr(hARuyObairWAY^G{bSQIlVRUtKJtsj-Oix2WR7p=xUsF^?P+w9-R8m<dENX9JUuk4LYHwmJb8K&PJ#%btbS!dZbairWV{c?-J#uApaBpsNWiE1MbairWV{c?-DGDGUARuLUV`Xr3ASh2$MRIa)ax8OoVsLVAV`X!5E>vl4Wp8zKMR;&&a%E&GAYpSLWpZ+FaykkiARr(hARr)eWps6NZXjf7V{|BUbYXOLb3G?POiWKhK~zakPG3`0MNnT-MO0E*CoF1jVqa-wJ!)@aEOTscbUkxyZ*(kWa&m8SUvznJWj%CxaAhcEa&m8SDK1}MZeeX@UtcK-3JPRpW*~QAbYo~=VQg$~V_|e@Z*Fraa&K>REOK>jbZKp6EO&2mYb<bibZBpGEOT^ka4ae;a&>NQWpX`pbz*RGZ)0V1b1rgqZYeqnARr(hA|fJBVsmA3c4Z)CcxiKVX>MmAVQg$~V_|e@Z*FrtAZ}%LWpW^CZggdGa&>TYAYpDGVPkY@Z*CxCXlZO@E(!`DARr)3X>)UFZf78JWpH6>av*JCa%*LBb0B1Oa%pa7AZcxIZ*p{VAY^YKZf|rTVRdw9Z*pmRWgt^YM@ddpE+A8MVRCeJa0(zGARuUOZ)<ZPVRB_4Y;R#?Wn>_3VRUtJVQg%9AYyqSWnp7zAZ}%MAYo&4X>V>jAarPHb0BwNbYo~`av*PRY<VDWVsmA3c4c!4ARr(hXkl_>cVTj6AaidZb8K&SAZcxIZ*p{VEFf-WcOYYIVQFo1EFf}abaH8Bb0A@EWFT^7b97;HbaNnMVQy}3bRc7LWnpw>AYlq1ARr)eWpHI-bY)~9V`yb#YjAIAZgeejWoB|^b7&xJZ*Ooe3LqdLAR;0n3LqdLAa7!GWpZ|9WIcO*J8WTdbYEd~bY*RDbUk~03LqdLAaHUnWo>D6C?`@eUqMw=PfbBoNkd;lNJ&gYUs6RzQbkipUqne$K|)MLL?<i?ARr(hARr(ha%E<6Wpij>aBpmBV|hI%L`hRYLQF+OUrt3=UqM4uNl#8wUrt3<MN(f=Nk>jWOiV>YCoBpeARr(hARr)fV`*h>bZKU3V_#!vX>4S2Wo}<{X=iR>Y;0v@Juob6X?A5_VQg$~V_|e@Z*Fs6b7^O8VQg$=WIZq`3LqdLAa`hKY-J#BZ*(AYbZ>AjX>(t5WppShItm~lARr(hARu&dc{&OpARr(hARr(hARr)WZ(?&jV|8+JWo~p|YHwn5C~|dfbZKp6DLZmyW@U0^ZewLEb8mHWV`V*ZWoBh^Wo~0-UvhPBbZKp6C~|LabSVlTARr(hARr(hARr)QWo966Ze=KGbY*QRItm~lARr(hARr(hARr(hARu#WZ*(kbZ(==ZbY*QjAZulLJt%W*Z*(kbZ(=DsAaHVTW@&6?J#S)jWpZ|9WG-iAbSQIlawv0bZ*(avdwnSiARr(hARr(hARr(hARr(hX=WgBa&Km7Y-KKIWppSfYHwm+X=EoUAw6nuVjypFAZ~ATAaHVTW@&6?E@x$QC?|Gha%pC1Wn?EQItm~lARr(hARr(hARr(hARr(hARuXGAarSMWiD-RZf|sNZfRpEDJ^Vab97%}baZ8HaC9zbWppTOWqB+uF=aU^K0P@wItm~lARr(hARr(hARr(hARr(hARr(hARugEb97%}baZ8HaCBR1WqDmabZKp6E^TjaZ**^NX=5lU3LqdLARr(hARr(hARr(hARr(hARr(hAaZ4Mb!>D!Zf|5|Ut?ixY$zvia&KZ~CoFPrZ*(kjb#8QNZDlNTY;SZdYHwmJaCvlSZ*DAdb#88Day@c&Zf<3ADGDGUARr(hARr(hARr(hARr(hARr(hARuRNZ)80va%FRMY;-PXWppSfb97;Jb#o^vJv}EuLsV5kOkYw}PE<)vMPE-sQ$<o%MMPgtPhUn!R3{1`ARr(hARr(hARr(hARr(hARr(hARr(hARr)NZe$>GWpj0GbS`IQbSNitbY*Z~a%FUNa&BX9WMwBQJv}fWVQyp~X>)0Ab97;DV`V6EWpj0GbS`IQbSNisb#8QNZDn6@a&Km7Y-J}YEM#e8bSWtcARr(hARr(hARr(hARr(hARr(hARr(hZ(?(0a&~28TXS@BD06IYbSYguWNBk`C~{?Ub!>Djc4cyDW@%+)J!fxkWGM<DARr(hARr(hARr(hARr(hARr(hARr)caxQRnbSQUka%(RqZf|5|EpT#gVr6qDFDNH-Y;SZeC1fonb1rIgZ*C_gD06IYbS!FbVks#sa%FRMY;-9KARr(hARr(hARr(hARr(hBOpX&Y-wU;a$$63Y<VDVZy<7IW^!e7Xdq)@Y-}zdK_G5rcOYYEX>4R5V_|L}Y-ML@bZKp2bY*OL3LqdLARr(hARr(hARr(hAR{1bVPk6`a%FH~X>u)LV{~bDWiD!SZ*CxUZggpEAZcxIZ*p{VAY*TBaBO9CWji2cc4cyTAY*J{X>AH1ARr(hARr(hARr(hARr(kAYpD~AYp85Z)0I}X>V>Ia%FIAVPj=&Wo~pJZFO^WAaZ4GVQFq4W^!d^AarjaW@&C|b7(FKARr(hARr(hARr(hcWHEJAY*TCV|8+JWo~pXW_5IRa%FQaRA_Q#VPsHmZ)`<)Wn*=8Z*nMYVR&D6Z*ps8a&tX1DIj5UAaHMQY&r@cARr(hARr(hARr(hARr)YX>)WaaBpvHE^T3OC~t0MENX9Jb15kbARr(hARr(hARr(hBOp>`VPs`;b0A`AZe$>3c4cyTAa7!GWpZ|5bZKvHAYx%-Yan!QAarPDAY*lMa%FCGAYp85Z)0I}X>V>INklFRARr(hARr(hARr(haB?nib#y3qZ*prdCum`EWOre5Wi4-Fb7gXNVRUJ4ZgVbbb8l`ZEM#e8bSQIUXk~3-Uv_13b7^mGJ#ca^NI_CWS3y!mUsFR!MNL603LqdLARr(hARr(hARr(hAaZ49WpZV1V`X1*b#8QNZDn6@a&Km7Y-K%iWoBh^Wo~0-EOKRLWpZV1V`X1+Z*_8GWj%9mb#h~6EOTscbaOp#VsmA3c4cHN3LqdLARr(hARr(hARr(hAa!tLVRU6=Uv+e2J#=YpWiE4ca%OaCZDlAYC0Q*cZ7n5aR3%6{B~3adQ(7l1bZKp6E@y3YX>DaFDJd)pARr(hARr(hARr(hARr(ha%E<6Wpij>aBpmBV|hI%L`hRYLQF+OUrt3=UqM4uNl#8wUrt3<MN(f=Nk>jWOiV>YCoBpeARr(hARr(hARr(hARr)fV`*h>bZKU3V_#!vX>4S2Wo}<{X=iR>Y;0v@Juob8Wp`g;Y;131VRUJ4ZgXE_a%Ew3Wn^Dsd0%jHZ(?OVMqzAoWhp5NARr(hARr(hWq4y{aC9I=cw=R7bZKvHAYpSLWpZ+FaykkiARr(hARr(hARr)caxP_UX>=$jQZZjeSV>b<Nlr&!K}<|fLqSwYPflM?LQ_RjRzXxrPflM_MO0E*CoE-ha&K~9ba`-PJ#=|+WhiBGa&K}eE?-}6VQpn!UnvS8ARr(hARr)fbZ>AjcVTICC^0oD3JMBjWo964VQyh(Whh~CXLBh!3LqdLAaZYSbUjdEbZ97Ha%Xcca&K>RDK2tlb8l>RWhf~-cW-iQJy2nEXeeQFXLBxhZ*prXJ92eybZKp6J#uevbT4pnE>BQJQbANnPfkHh3LqdLAa87BJ#TVpXK8L>Y+rC;V{2h&WhioQZ*(kSa%XccXm4$0DLZp?Z*V<yXmVv?WNB_^E=6`_ZgeOq3LqdLAZBlJAaiMFAZczOD069NZeeULQ%OfuMN&;Hb7^O8VQel_Nk>UeR4F=hX=iR>Y%X(YXKrC^D069NENo$IVq{?;DqlKtbZ>Ajb7gcWDJcpdARr)jX>@2HaB?neZ)0mHcW-iQFDGqbZeeF-axQFdV{0cYVr*|?YiVw0Jw{<{b7d(y3LqdLARr(hAaZYab3JBhZe(9$Z*FvQZ)|L3a&sthZ*O!gZ){{K3LqdLARr(hAZBlJAaZYaAZczOa&LEYItm~lARr(hARr(hARuXGAaZYaTPJX7WG7ucJ#TX^XJvG7X=ErVI%98cbZKsNWeOl5ARr(hARr(hARuXGAaZYaE@x$QC?{iea&l#EbYE~`bYo~IDLQgtX>(;DQgv>0X>Daia&m8SC?`QqPgF=nQeRR<P(evjMMPgiPfk=)PfScjQeR0^Us6?0PDxHjCn*XbARr(hARr(hARr)caxQRnbSQUka%(RrCvR+IEn{zPbaHQOY-MsSC1fsYb8l`ZC30_fTPJX7WG7uIEOT^ka9?k1WM5-%Zgg^QY;0w6C~|LiEOKvebS!UdWGN{MARr(hARr(hb#8QNY&~>oZDlTPZ*FgNZ*FO0C@Cv3H3}dgARr(hARu>WX>4U6Y;R+0Uu<c1WhioWZggpFWiKaVZ*FvQZ)|L3axQFdV{0cVAYpD~AarSMWiD-RZf|sNZfRpEDLi#<bZKll3LqdLARr(hARr(hAZcbGb98TTE_Y#RbSN${DLQgxbairWAU8D%ARr(hARr(hX=Wg7Z)0m;Y-x67C~|dfbZKp6FDGMfZgg^QY;0w6E^KdOYbPl>a$#w6Wgt>@ZggpFWkqswZ*nLnPfSE#Lr+drQcp}wMN(f(PeV&zPES-{QbkNfK~qIUCn*XbARr(hARr(kAW~&>bZ>HHAarPDAa8PMXK8L>Y#?uNWpZJ3X>V>}Y#?cEaBp&SAaG%HXdq%`W^Zz3AarPDAYpTPZewV2Z*FgOa|$3JARr(hARr?kaB^>AWgv28VPs`;Aa!$Pb0BGSb0B4SVPkY4Xkl_>cVTj6En{q9b8{eLVQz13X=7n*X?kUHE(#zZARr(hARusZE^KdMWM6M^WpZJ3X>V?GC~|LabS!UdWGM<DARr(hARr)jVRU0?WpX`qXmVv?WNB_^E>vi8Wnp9}bYXI5Wpq7vVRU0?Utw%)Z)0I}X>V?GEManIb3G_>Z*O!ga&>NWX>DaJcW-iQEManIb1rasbZBpGEOT^ka49TgVP$P^Zaq|Tb!90EARr(hARr(hcVTp6Xk~IPb97;HbSNnbARr(hARr(hbaHt*3LqdLARr(hARr(hAaZ4Nb#iVXaB?nVZ*FvQZ)|L3a$jt3Z*VAbZ*O!gZ){{NcW-iQEOT{YZE18pVRC14E^~EaZE18Yb98TTJ#%z#a4c+Lb#7y5UvzJ4Wo|uTa%XccY+-e7V`yJ=Z);_4DGDGUARr(hARuOGZeeU}c{+1+Z*VSiWppShJ9lApV`yb^E^2RSZYXqVZDntDbUiLLDGCY-WMyU`b97;HbSPnRXLBqnEO2jdWo|ulbz*RGZ)0V1b1qPCaAj^RV{C6@Ydv&nZDlTPZ*FgNZ*FO0EOTsSWpF)oX>DaLb8Ka0a49+pARr(hX=Wg9Z*(AGa%Xccb9G{EX>>YrVQF(^AXZ^)b!A0za&K}dCq-CLOi4pYR9{n7LQP3jUs6R;RY_7sL?<Z<ARr(ha&K>RJy2nEXeeQFXLBxcZ*O!dE^=jaZ)|pDC@DK|VPk7yXJtK5VRUFHVRC14E^uLEYhh<)DK2tlb8l>RWhf~MARr(hcW-iQJy2nEXeeQFXLBxhZ*prOZ*m||VRUFHVRC14E@*FUWhpNwaA9y|ax*P*F)e0wY-}xZWpH6>axZspa%(3kE^=jaZ)|pDC@DL4Z*prcZEIv{awu?Na%FCGb3IgYb!9ANcxiKVUvFzYRC0A?DGDGUARu>XbZ8)OaxQFdV{0gPZ*prdCv$XRa&#_iZ)0mGDLM)uARr(hARr)NaBysCWn?{RZgX^DY;0d>Z*OaJE@5zRY<VbhZ*O!gVRC14E@*FUWh`)EV{2h&Wj%0VV{2h&Whn|EARr(hARr)bY-Bxea%pF2ZeeU+aA9L>VP|D1a&K>REManIb1rCaZDlD6ARr(hARr(ha&LEYJ!WZcWM5-%Zgg^QY;0w6b0~6eZ*(kgY-A}4ARr(hARr(hWq4_GbZKs9J#J-qbSNltAZBlJAaWpSZXj}RcXJ?VW*~AdXJvFKCu4PTa%FCGUvOb`V`wKSDJ)KJZe=M7ARr(hARr(haB^jKX>WCNJ#ca^a%Ev;Uutu2ZYXzea%(RqV{dMBa&K&GWpXWWa&Kd0b8{|gb8l`ZEPH(^Aa8OYdwmKZARr(hARr)VW*}vFX>)XGZf782Ze$>Ea%FaDZ*_AnXJvFKCva(GCn-HWWq4_GbZKs9TPJX7WG7u9VQyp~aB^jKX>WCNE@x$QC?{-Tb#7y5UvzJ4Wo{=aJw0!5bZKvHC}ntQb98BLXIm#>a%XlYT`VUpEo@<RZewUIbZ={AZYL=^3LqdLARr(hARr(hAaZ44WO+SsaxQXZVPs!wb8l`acW-iQFDGMfZgg^QY;0w6EplaHWO*)Xb8l`ZEPH(^Aa8OYdwmKZARr(hARr(hARr)TZ*OEhC~{?CWO*)UWppSfaA{;GDLp-9cxiKVX>Mm*Cva(GCtV<6Ze$>GWnpA_E@x$QC?{-Tb#7y5UvzJ4Wo{=aJw0%8Wp-(Ab#q%MY+-e7V`yJ=Z);_4CtV64ARr(hARr(hARr(hARr(hAYpD~AaZ44WO*)UWppSfW@&PBbYF99b#iTAZ(?(0a&}>KX>V>{V{dJ6Y-Mz1WG5*gX>%Y{a&=`2ARr(hARr(hARr(hARr(hARu9GWFT^7VPttOXJvFKCvkOUb!A^~VsmA3c42gBZ*E_6bz@^?b8}{OY$quoX>%Y{a&=`X3LqdLARr(hARr(hAaZ4Nb#iVXWNBk`D06gSbait*Cq-CEQ&dS#M_*DgUs6R-K}k|yLr+drQcp}wMN(f<MO9NpL?<9=W*}#8Z)6~4Y;$EGCq-CEQ&dS#M_*DgUs6R-K}k|yLr+drQcp}wMN(f=R6$ZyRZw40MNULXPDdv!3LqdLARr(hARr(hARr(hARr(hARr(hAaH49J!N=lb98BLXIm$5X=Eo|EO&2mYdv#xawvCia%(9pbY*gFX>MU`Ut?izUt?@<b7ehLa&=`X3LqdLARr(hAarkQWo|uaVRL9~X<{yOXkjunHYjg%E_HHYZe(w5C^IrCDK2Pbcw}j3Wpi{WDGDGUARr(hARuFJZEaz0WIbD9a%XccaCvlSZ*D9nElDRVCoNMaEGI2OCoCr|btf!ybaE(2MN&mCCv$ahWpZ|Db8m7kaCs*wEGKPYZeeF-CoCr|Epl&fbSErxbaE(iZ*O!dED9hXARr(hARr(hARs3#Eog6TWhX3ibaE(Ra%XccXm4$0DJ&-~EpTCDYhh<)CoFSxawu?NV{2h&WhpEtEiHF%a%(3nb98bjcW-iQDJ&-~Epv5ZZE18TEGI23Y+-e7V`wdOZ);_4CoFVtYh`XMCoL^-d30!RZYL~Za%XccaCvlSZ*E-*ARr(hARr(hcWHEJASicla%(RqV{dMBa&K&GWpXZTZ)YbdE^lyUZYU>VVkazOb!KK|a%pa7JuoRCVRImCZ)Z9RARr(hARr(hARr(hV`yn?WIb?iaAj^NV{dJ3VQyqBb97{BZas5#VsLVAV`X!5E<{CEPE|}yEOT^ZZ*_D%Y;R{Qb97{7a&kRvZ)YrXbYXIIUv6c0Uvp)1b7^mGJyddaWh`TCZ*yf|W@K|cRC0A?DGDGUARr(hARusZE^u{pD0gpiYcD5bZ*FvQZ)|L3axHLjZ)0V1b1rIgZ*C_nWNBk`C~#?HJ!5ETY-BEQX=E&HVRdd}XkT=1Yh`XdbZ={AZY*+dZ*)C#baE(iZ*O!dEOT^Wa&%>6Uv+e2J#=YpWiE7SZDlAaDJcpdARr(hARr)fbYXOLb3G?hR6$ZyRZw40MNULXPDdv@b#8QNY&~OaZ)0mHDJwV%ARr(hARr(hcW7yBWgt{?b!9pVARr(hARr(hARr(ha%Ev;c|CA)E^=jIWM67?Z*C}eZ*prdCu47JbaHQOY-MsSa%Ev;c`j;mZ*C_ndwnS&Z*m}eeF`8TARr(hARr(hARuXGAY*80Y-BEQZ)|KRDIjTcAZ~ATAWm;?Wjb?oVRUtKJttFCK~hvzP+vwtNlZmVCp%(tWnpUyARr(hARr(hARr(hX=WfOa%Ev;c`j#VbSNiqX=EoUJw0P+X>4RJaA{;9VQyp~a%Ev;c`j#VbSNimVRdd}XkT=1Yh`XHDLp-OZ);_43LqdLARr(hARr(hARr(hAYpD~AaZ44WO*)UWppSfW@&PBbYF99b#iTAZ(?(0a&}>KX>V>{V{dJ6Y-Mz1WG5*gX>%Y{a&=`2ARr(hARr(hARr(hARr(hVQyp~a%Ev;c`j#VbSNirb!ByBUvFY_WpZ|5bZKvHUvqV1V`X!5W_4^QDIjTcAXIX7WhpvybYXOLb3G?gF<(+eP(evjUqeq$R8mh&Ohr;(Qbj>TStmPUa%Ev_3LqdLARr(hARr(hAZcbGV{C6@YbYr`J#}t$X>2-Ta%Ev_3LqdLARr(hARr(hAaiVGWpF4iF)0clARr(hARr)iVQh6}J!ENPbSQIlVRUtKJ#%znbaitqaA{;cV`yn?WG--NWGr`Ya%(+vbaE(nZ*prXEOcdZZE0>{Y+qwxZeL?;Z*yfmRC0A?ED9hXARr(hARr(hARu9EY;R*>bZKvHUt?izV`Xe?VRUJ4ZgV{_EOTRNWo~q7W@%$zW@&6?b6;a<VQyz-WIZq}V{dJ6Y-Mz1WM6V+b9HQVb6;|0b7gcrFf0lnARr(hARr(hARr)fV`*h>bZKU3V_$M*W^!e7Xgw!ycWz~5AY*80Y-Au)Nk>UeR3Kw$Wn*h_Z)t9HAa!ncX>Md7Z*FXPJ0NavAX7<4OG!*jCoBpeARr(hARr(hARr)cVRB(?Y-Ma|b8S5*X>Md?aAj^}Wo~pJVPkY@Z*FrSZ*CxMWnpu5a%E&7V{dJ6VRUI?Y-J#GY;SaPJ0Npqa%o{~Aai4BWo~q7W@%#}VqtP}X=QSAAaZ4NVQFq<WG5^NARr(hARr(hARr(hXm4+8b6;U_aBOL1WIbVUaBOL1WGM<DARr(hARr)caxQRnbSQUka%(RqQ&d4xR7FHCYIARHCoFbhY;|QRJ91@ob#iVXc42IFWeN%kWMyU`ZDDC{C}DDEc0EpSZe=Mt3LqdLAaFfla%XU1a&u)aL2_qxZDnqBP+@X&WpXHCY;13LUtwZma%FZsMqzAoWhpywE@5P3Utw}*b!}yCbSNinZ)9aBEMsVIX=7z`Jt!x0bYXIICoCspXk}w-CoCszVQyh(WhW^q3LqdLAaE{WWMp4qa%XjIWo~pRCoL^<Z*O!bEOdErWj#<~bZ9JOWoBV@Y;-+PVRUFHCogbyVr*$+FKBOVWiNVYb$V%ebzyEVaB^>IWn*-6FK}USWpXnoDJcpdARr)cE@5P3Utw}*b!}yCbSNh+Eog6TWhX3jd2nStP+@dvEM#S7VRdYDJy2nEXf9}PZDlAaDLZg3VPs@qVRC16ZDnqBC?_o~aA9L>VP|D0EOdErWj#<~bZ9JOWoBV@Y;-+HMN&m63LqdLAaE{WWMp4qa%XjIWo~pRCoL^^Z*prVEOdErWj#<~bZ99%a4unFWM5%&XLW65ZgeOoEiG_)bZBpGCoE)TW?^+~bUkx<b1r3gWn*=8VPb4$DGDGUARurqVPs@qVRC16ZDnqBC?_o~b9G{EX>=znVPkY@Z*Dy&b98TVWnXl1b!8_hJ8&*xWMp4qa%XjIWo~pRCoL^(VRdd}Xf1SaYh`XHDLZg3VPs@qVRC16ZDnqBC?_o~a%E<6WpijPaB^jIa$#w1X>MmHEMa4GX>V>lCv$Xfa%Eq1a&=`VDGDGUARu9KXLCJpE^uLTb7fy)a%XcWVRC17DGDGUARuXGAYpQ6b1rRfWMw@)Cv9PFVP|D0Itm~lARr(hARuyOb9HQVJ#ArbVP|D1VRC14DGDGUARr(hARuyObairWAaZ4Mb!>DXX=Wg4b7^jKbYX5|WhioGb9HQVENO0ZDIjHRb7def3LqdLAa-GFb!9zsbYXIIC}DDEb15KcW*}j5XLBxXZ)9aXJtuQ?VRCdQAZ2WGWguy8b97;BY+q<^Z)<ZdX>N0HWn*+GVRC14E^=>gbSz<VXLBxSZ*65M3LqdLAaHVNZgePWb8l`gWOZ$Db0}nKV{|BGc4cmKJttBzUqeq#P)tQsMPE`yP(evjCoC!|c42IFWhpFWZgX{VWnW=)V`*tUMqzAoWhpFXY;|*JJyddaWhn|EARr)eWps6NZXhxsX=WgHVQh6}E@x$QC?|7tVRUtKCn-HWCsR~GQdCt?Uq(SmOhrT|AZ2WGWgsvL3JPgvAYWf@VQpn!Up+l1UteuuX>MO%CpvOrX>(;DQ+acAWo<=xX>=%UVQFqCDJcp'
SUPERVISOR_SHA256='43aeea5f1714c8171352b3ec2c86a2e907c8b49d95431b96da33fe8b9f8db0c7'
DEPENDENCIES=('parallel_runtime.py','install_hooks.py','atomic_io.py','newnode_tool.py')
RUNTIME='runs/training_campaign_parallel_v1'


def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()


def read(path):
    with Path(path).open('rb') as handle:
        size=os.fstat(handle.fileno()).st_size
        if not 0 <= size <= 1024*1024:raise ValueError('HOTFIX_RECORD_TOO_LARGE')
        data=handle.read(size+1)
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('HOTFIX_RECORD_NOT_OBJECT')
    return value


def optional(path):
    try:return read(path)
    except FileNotFoundError:return {}


def regular(path):
    value=Path(path).lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid():
        raise ValueError('HOTFIX_FILE_NOT_OWNED_REGULAR')


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():regular(path)
    fd,temp=tempfile.mkstemp(prefix='.refresh-hotfix-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)


def option(args,name):
    if args.count(name)!=1:return None
    index=args.index(name)
    return args[index+1] if index+1<len(args) else None


def identity(pid,proc_root=Path('/proc')):
    try:
        if type(pid) is not int or pid<=1:return None
        folder=Path(proc_root)/str(pid)
        if folder.stat().st_uid!=os.getuid():return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return None
        argv=[v.decode('utf-8') for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid,uid=folder.stat().st_uid,start_ticks=fields[19],argv=argv,ppid=int(fields[1]))
    except (OSError,ValueError,IndexError,UnicodeError):return None


def checked_manager(record,root,home,work,old,overlay,*,process=identity):
    pid=record.get('pid');value=process(pid)
    if value is None:raise ValueError('R1_MANAGEMENT_PROCESS_NOT_FOUND')
    args=value['argv'];at=1
    if not args or Path(args[0]).resolve()!=Path(sys.executable).resolve():
        raise ValueError('R1_MANAGEMENT_PYTHON_DIFFERS')
    while at<len(args) and args[at] in ('-I','-S','-B','-u','-E','-s','-O','-OO'):at+=1
    allowed={old/'supervisor.py',overlay/'supervisor.py'}
    if (at+1>=len(args) or Path(args[at]).resolve() not in allowed or args[at+1]!='manage'
            or '--child' in args or '--controller' in args
            or option(args,'--root')!=str(root) or option(args,'--home')!=str(home)
            or option(args,'--work')!=str(work) or option(args,'--package')!=str(old)
            or option(args,'--launch-token')!=record.get('launch_token')
            or not record.get('launch_token') or record.get('root')!=str(root)):
        raise ValueError('R1_MANAGEMENT_IDENTITY_DIFFERS')
    if record.get('start_ticks') is not None and record['start_ticks']!=value['start_ticks']:
        raise ValueError('R1_MANAGEMENT_PID_REUSED')
    return value


def verified_files(old):
    regular(old/'MANIFEST.json');raw=(old/'MANIFEST.json').read_bytes()
    if sha(raw)!=OLD_MANIFEST_SHA256:raise ValueError('ORIGINAL_REPAIR_MANIFEST_DIFFERS')
    manifest=json.loads(raw);files={}
    for name in ('supervisor.py',*DEPENDENCIES):
        path=old/name;regular(path);data=path.read_bytes();record=manifest['files'][name]
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:
            raise ValueError('ORIGINAL_REPAIR_FILE_DIFFERS: '+name)
        files[name]=data
    if sha(files['supervisor.py'])!=OLD_SUPERVISOR_SHA256:
        raise ValueError('ORIGINAL_SUPERVISOR_SHA256_DIFFERS')
    replacement=base64.b85decode(SUPERVISOR_B85)
    if sha(replacement)!=SUPERVISOR_SHA256:raise ValueError('FIXED_SUPERVISOR_SHA256_DIFFERS')
    compile(replacement,'supervisor.py','exec');files['supervisor.py']=replacement
    return files


def overlay_install(overlay,files):
    if overlay.is_symlink():raise ValueError('HOTFIX_OVERLAY_IS_SYMLINK')
    overlay.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name,data in files.items():
        path=overlay/name
        if path.exists() or path.is_symlink():
            regular(path)
            if path.read_bytes()!=data:raise ValueError('EXISTING_HOTFIX_OVERLAY_DIFFERS')
        else:atomic(path,data)


def ready_for(work,pid,token):
    value=optional(work/'controller-ready.json')
    return (value.get('pid')==pid and value.get('launch_token')==token
        and value.get('first_slurm_observation_completed') is True
        and value.get('queue_observation_successful') is True)


def manager_or_stopped(record,root,home,work,old,overlay,*,process):
    """Recover only a management start/stop durably recorded by this hotfix."""
    if process(record.get('pid')) is not None:
        return checked_manager(record,root,home,work,old,overlay,process=process),False
    before=optional(work/'refresh-hotfix-before.json')
    own=optional(work/'refresh-hotfix-applied.json')
    if (before.get('hotfix_id')==HOTFIX_ID
            and before.get('replacement_supervisor_sha256')==SUPERVISOR_SHA256
            and before.get('old_supervisor_sha256')==OLD_SUPERVISOR_SHA256
            and before.get('record')==record and isinstance(before.get('identity'),dict)):
        prior=before['identity']
    elif (record.get('hotfix_id')==HOTFIX_ID and record.get('supervisor_sha256')==SUPERVISOR_SHA256
            and own.get('hotfix_id')==HOTFIX_ID and own.get('supervisor_sha256')==SUPERVISOR_SHA256
            and own.get('manager_pid')==record.get('pid')
            and own.get('launch_token')==record.get('launch_token')
            and own.get('start_ticks')==record.get('start_ticks')
            and isinstance(record.get('argv'),list)):
        prior=dict(pid=record['pid'],uid=os.getuid(),start_ticks=record.get('start_ticks'),argv=record['argv'])
    else:raise ValueError('R1_MANAGEMENT_PROCESS_NOT_FOUND_NO_RECORDED_HOTFIX_STOP')
    checked_manager(record,root,home,work,old,overlay,process=lambda pid:prior)
    return prior,True


def refresh_steps(manager,root,old,*,process=identity,proc_root=Path('/proc'),orphans=False):
    """Identify only this R1 CPU refresh srun, never a fit or an allocation."""
    runtime=root/RUNTIME
    state=optional(runtime/'scheduler-state.json');jobs={}
    for slot in range(3):
        attempts=state.get('slots',{}).get(str(slot),{}).get('attempts',[])
        if attempts and str(attempts[-1].get('job_id','')).isdigit():
            jobs[str(slot)]=str(attempts[-1]['job_id'])
    found=[]
    for path in Path(proc_root).iterdir():
        if not path.name.isdigit():continue
        value=process(int(path.name))
        if value is None or value.get('uid')!=os.getuid():continue
        if not orphans and value.get('ppid')!=manager['pid']:continue
        try:
            if int(value['start_ticks'])<int(manager['start_ticks']):continue
        except (ValueError,TypeError,KeyError):continue
        args=value['argv']
        if not args or Path(args[0]).name!='srun':continue
        for slot,job in jobs.items():
            expected=['--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',
                '--mem=256M','--gres=none','--time=00:04:00',sys.executable,'-I','-S','-B',
                str(old/'newnode_tool.py'),'refresh','--root',str(root),'--runtime',str(runtime),
                '--job-id',job,'--slot',slot]
            if args[1:]==expected:found.append(value);break
    return found


def same_step(current,previous):
    return current is not None and all(current.get(k)==previous.get(k)
        for k in ('pid','uid','start_ticks','argv'))


def apply(home,root,*,activate=False,process=identity,send=os.kill,popen=subprocess.Popen,
          clock=time.monotonic,sleep=time.sleep,random_bytes=os.urandom,scan_steps=refresh_steps):
    home=Path(home).expanduser().resolve();root=Path(root).expanduser().resolve()
    base=home/'paper3-r1-full-repair';old=base/OLD_PACKAGE;work=base/'work'
    overlay=base/('refresh-fixed-'+SUPERVISOR_SHA256[:20])
    if base.is_symlink() or old.is_symlink() or work.is_symlink():
        raise ValueError('HOTFIX_BASE_PATH_IS_SYMLINK')
    files=verified_files(old)
    regular(work/'controller-process.json');record=read(work/'controller-process.json')
    current,stopped=manager_or_stopped(record,root,home,work,old,overlay,process=process)
    result=dict(hotfix_id=HOTFIX_ID,old_package=str(old),overlay=str(overlay),work=str(work),
        manager_pid=current['pid'],scientific_children_signalled=0,allocations_cancelled=0,
        completed_results_reset=0,scientific_files_changed=0,old_package_files_changed=0,
        automatic_child_refresh=False)
    if not activate:return dict(result,status='READ_ONLY_HOTFIX_READY')
    with (work/'refresh-hotfix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Re-read after locking: no stale PID or overlapping installer may act.
        record=read(work/'controller-process.json')
        current,stopped=manager_or_stopped(record,root,home,work,old,overlay,process=process)
        own=optional(work/'refresh-hotfix-applied.json')
        pending_identity=(own.get('start_ticks') is None and record.get('start_ticks') is None)
        if (not stopped and own.get('hotfix_id')==HOTFIX_ID and own.get('supervisor_sha256')==SUPERVISOR_SHA256
                and own.get('manager_pid')==current['pid']
                and own.get('launch_token')==record.get('launch_token')
                and (own.get('start_ticks')==current['start_ticks'] or pending_identity)
                and str(overlay/'supervisor.py') in current['argv']):
            if pending_identity:
                record['start_ticks']=current['start_ticks'];own['start_ticks']=current['start_ticks']
                atomic(work/'controller-process.json',encoded(record))
                atomic(work/'refresh-hotfix-applied.json',encoded(own))
            return dict(result,manager_pid=current['pid'],status='R1_REFRESH_HOTFIX_ALREADY_RUNNING',
                controller_ready=ready_for(work,current['pid'],record['launch_token']))
        overlay_install(overlay,files)
        saved=optional(work/'refresh-hotfix-steps.json')
        captured=(saved.get('steps',[]) if saved.get('hotfix_id')==HOTFIX_ID
            and saved.get('manager_identity')==current else [])
        requested=set()
        def collect_steps(orphans=False):
            nonlocal captured
            rows=scan_steps(current,root,old,process=process,orphans=orphans)
            for row in rows:
                if not any(same_step(row,prior) for prior in captured):captured.append(row)
            atomic(work/'refresh-hotfix-steps.json',encoded(dict(hotfix_id=HOTFIX_ID,
                manager_identity=current,steps=captured,scientific_workers_signalled=0)))
        def stop_steps():
            for row in captured:
                signature=(row['pid'],row['start_ticks'])
                if signature in requested or not same_step(process(row['pid']),row):continue
                # The captured exact srun command launches only a metadata
                # management refresh helper on its own 256M, --gres=none step.
                try:send(row['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
                requested.add(signature)
        collect_steps();stop_steps()
        # Recheck identity immediately before the management controller signal.
        if not stopped:
            check=checked_manager(record,root,home,work,old,overlay,process=process)
            if check!=current:raise ValueError('R1_MANAGEMENT_CHANGED_BEFORE_SIGNAL')
            atomic(work/'refresh-hotfix-before.json',encoded(dict(record=record,identity=current,
                hotfix_id=HOTFIX_ID,old_supervisor_sha256=OLD_SUPERVISOR_SHA256,
                replacement_supervisor_sha256=SUPERVISOR_SHA256)))
            try:send(current['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            until=clock()+12
            while process(current['pid'])==current and clock()<until:
                collect_steps();stop_steps();sleep(.1)
            if process(current['pid'])==current:
                raise RuntimeError('R1_MANAGEMENT_TERM_PENDING_NO_WORKER_SIGNAL_SENT')
        # A launcher can be created immediately before its old parent exits.
        # Adopt it only through its exact old helper/root/current-job argv and
        # a start time at least as recent as the verified old manager.
        # Re-scan even when the first scan is empty: a launcher forked just
        # before shutdown can still be completing exec at the first read.
        # This short management-only settling interval leaves scientific
        # workers running and covers the ordinary fork/exec handover window.
        settle_until=clock()+1
        until=clock()+12
        while True:
            collect_steps(orphans=True);stop_steps()
            pending=any(same_step(process(row['pid']),row) for row in captured)
            if not pending and clock()>=settle_until:break
            if clock()>=until:
                raise RuntimeError('OLD_REFRESH_STEP_PENDING_NO_NEW_CONTROLLER_STARTED')
            sleep(.1)
        # PID reuse is not ours and must never receive another signal.
        token=hashlib.sha256(random_bytes(32)).hexdigest()
        command=[sys.executable,'-I','-S','-B','-u',str(overlay/'supervisor.py'),'manage',
            '--root',str(root),'--home',str(home),'--package',str(old),'--work',str(work),
            '--submit','--launch-token',token,'--python',sys.executable]
        with (work/'controller.log').open('ab',buffering=0) as log:
            child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                start_new_session=True,close_fds=True)
        fresh=dict(pid=child.pid,launch_token=token,root=str(root),started_utc=time.time(),
            argv=command,hotfix_id=HOTFIX_ID,supervisor_sha256=SUPERVISOR_SHA256)
        # Persist this exact child handle/nonce before /proc can lag its exec.
        # A failed or interrupted startup can therefore be recovered on rerun.
        atomic(work/'controller-process.json',encoded(fresh))
        receipt=dict(result,status='R1_REFRESH_HOTFIX_STARTUP_PENDING',manager_pid=child.pid,
            launch_token=token,start_ticks=None,supervisor_sha256=SUPERVISOR_SHA256,
            old_manager_pid=current['pid'],management_signals=['SIGTERM'],
            old_refresh_step_pids=[row['pid'] for row in captured],
            management_refresh_steps_signalled=len(requested),
            old_refresh_steps_remaining=0,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        until=clock()+2;new=None
        while child.poll() is None:
            try:new=checked_manager(fresh,root,home,work,old,overlay,process=process)
            except ValueError:new=None
            if new is not None or clock()>=until:break
            sleep(.05)
        if new is None:
            receipt['status']=('R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED' if child.poll() is not None
                else 'R1_REFRESH_HOTFIX_STARTUP_PENDING_IDENTITY')
            atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
            return receipt
        fresh['start_ticks']=new['start_ticks'];receipt['start_ticks']=new['start_ticks']
        atomic(work/'controller-process.json',encoded(fresh))
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        until=clock()+8
        while child.poll() is None and not ready_for(work,child.pid,token) and clock()<until:sleep(.1)
        if child.poll() is not None:
            receipt['status']='R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED'
        elif ready_for(work,child.pid,token):receipt['status']='R1_REFRESH_HOTFIX_CONTROLLER_READY'
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--home',type=Path,default=Path.home())
    p.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    p.add_argument('--apply',action='store_true');args=p.parse_args(argv)
    value=apply(args.home,args.root,activate=args.apply)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return 2 if value.get('status')=='R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED' else 0


if __name__=='__main__':raise SystemExit(main())
