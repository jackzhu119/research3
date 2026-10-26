#!/usr/bin/env python3
"""Install narrow R1 clock/driver resume hooks; preserve native training files."""
import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

PAYLOAD_B85 = 'c-pl+Yj@i=vfuBoVAYdDDoe?d-%*x(uA8{6w~3v#o$kZd;h{*#W=)YQNjdgr{on6A01_Z6*=_f_yRk$9gTY`hZ@^!Bvn(r%G7px1oEBw}B)!R1FCY6+9^Z@H53|KGD2L0(o{LYzBn$8ST!b0Ce8k67w<jLrqKwlUKZ<WeQTn$cSwh$1HrU?V?@fB+?Pz;{^k#3iHy(%c{n6okcfJ=Mg!`lI(LCHa41?|6(RKhY4n{(R+k3mwZa5Y@^TTj{xVt-op&4Y+n|$r1StJ01c`k}uL}1dprvFaTQsnt+Sz1||JX(PT7{>EhMAkAF1DIi)0*+RA8z+$kbB!Zm#bseB>@Apk76xUUr9*~zoyLo0mX}tL-z<Z?5Y>a*ptw!qS@r#wB1@|;S#h1t^K4-O;jon@pMQd{Rih%lNrR;NTE+{2j{iksZsj6~9PHhnT&EWN<f2^VDGf7BvLGrP1jbE<*xE0}L+SMUHH>VDGjRa*d)JyL!Ht4d=8qEz4_{v4Kf@UMJUG<sPva>!6c3>QYC1m$$x0mOd6v7@3AUrpeFfw+OEuJ&c@~PIaF%i8(icAm2J;AAdkY`feTB(9OCpi;yQ*n_3%C)x_hqA4BvIix+<40d9xZKS(!;!1CUNQ5ecK)Pug23MfA1T#`)aS^$eu8mAwq$kz^w4pwU<kL7pJAe!vT=Ki$mq5U&i5GVZM9=ZyC7yA+!X5=N6->Ya?Cl$<@O!g544cM<CS-&(6g%2|{7_Vg9-G0IX<LlM5G75~o7VZ}wP<0@0-T%<Svy5p;{^*2DBk=N8%vPQFT!ja=)OY~~~RizF*u>-E}oa5a|qp_7M8=~{Pjl6YA$#NQPIGHmWIzaWxDmfQ=c|CWdJaOwD?BtGDZ&&+Y~Y4b^8rCCY5WQn8@u!)8uy^nKX_!|Lhd-37(`A7e+XW#iJ?`_uxCi=$XktX#@R7;Jkv4Taj|9*6G>7O3|<I;ck<H-l?3511R1g}gC{6$<Wf-<~?CJfh}@b4#vk9*CNGvBhmWci)Q2XdS($3|KYUPVLPqV_BN)~~mlhGF)_t9N((94G>>c&c*vYJwK9336Z60HNFNV2m`t&l8guLDB&M>s{+eY<+C&XE{wbybaPD0i)D?NiqVn1i45{PmXwH(_<eu2ylJ%l)oDjSj4HrFc?Jt&)yhur)QqW4<~Kx;`Ba9;>fQD(%=Rmt<ar#bj$2cq#li4KeeYWW0?f;!lx$o)IbaQDlSO5rJ-;%fWVo3M6l6<fEX|7$uKU0Bwa0>{-!w~8an>sVJX5=L^bO$0>Cs<i40j!YKQ^nlf>xvhqvN^m8#Q+o@FT(%M!X0LJAb~T$5zXW@>nSnGr>g8%eq~DCVe_{M(*h*+l7Yyc$PZSnSIz$%F@fqz~}YWJ&x%IiU4WT;5`Htxc2BJDMQ7$QF|8p4|;{63j7O`v70Sip@dO%CRJ!MC7l^(BS)QR*3u_S}VmTcR;1iv~>Z6;HmaktREVLuiv7w0)m4WglWPThnTT`Pa12gCT+fBT{I=lV*d*S+CSj`4|czES?s}pP_}s+uu(wpGy?(}G1Uh1I1#wcxSADnbpmemm;677;#~DtdZc4CT6x)wgz59kD#Zsc1Mm}&B`zOpx!%{MFsMQ~`j9DzDqmM5p@^4KUAA|Fnz%39H6@{Z3=+c+W}qThrQr3mheI1_tQx@BWwAk=$aX$z3ww>aXd%<Lf;6&7Dz*`ol;^Kql1vTRu`z(QH=yzh59zn1%>n4QWvjPMaTL$1jz|z>Ag8&SDd^#QAvL3H+&7ziYaOMJ6(s9R9)mPYt=Xeh-h%Z&`xDKG5M(Rt0|E$cKnXB`09H^m!3p#i;0=g$a1_EIu~1FAusmV1iqIiA`+RwP?jK!V9)J9F>Ayd@_;hsn?#JWzLlcxf!P>4|Aani2lX7Fy-XCpa<=yA=^W#%#s`x)0f4bD|6{eA|hVmr^{9s<9pGI=rZ4XoJbJMDES-4KfM$~ltgyokSP2t90rBvo;{Y`^qD@1CwjDYwBIgyEgWiSD)Yt(;g?L?o=FOBcIHXOC8nT+>={_2IxBPXw<0I{VGr{7`#2X~d(%^((8tf1GU;!BJc_`9=@pFSL49@}hVbaYe>t504CQU$sDGGB@Qe~+1}L9$%YA627LB`sT6=NfI1f*Sy%*|l$zpeSFU5y6fC&rXdYfxE2JHJ)^Jd(SmYJzXYBi4Oo*)M?3mbuUhOWu_zz(np78K!m>ot)pb?9s6i^D6$b(S}bgi^Sio3{j8Mk-L!71FrntubgfG65Wq@<Ww~jh_%i_`z?P)Qfj)<Ct#c9HittX8WtE_e7r@^!j1Iz(A%MUXGX5~Mj;tlvp+W}+Opz8b65NVbISNATAkiAA$x)D^uokPL1loutHKH{eJ|p{STmDvw8Lo12OE#&9KPpjYA`BM7mqRt^y++#oF|jI#GUe*i|EFj(=(t&?k-^@9;AUB7^jI&%{vTx6=H+{tB*U186j*YHT%7!Ha(da{6lC~xetda;baE>3JU{;a{P^NW`2Dxf#}}8!7<G_XOl$_%MyJCt-^@wHY6+4jE<jAsV>ZJKQ903k-46I#03<}mH%U-Ie;*`;zyU#?EHm)7P!JVijJtjwhs1wHWgT+!Co^DpcEQm?EpYggc;pY4k4yPT3nhYeW>I+TcgAwa86ikllkxs&irlL?^rC@HpF>O~pdz~seLqFs@w@1>X@_s1p%$TA0EbpKFGZ4bPsNmnNBIp%?X>(v&z(pVVIGq=<oSLCO4IM_9fv^_`GM@?)Q*+y=Hk~?3>K|NM#9@Q&0!V;)Qe{8jU%vHMsi_P$PI6^7*v7hjNGxi-S311-3w2GqhAQHZ~&_u9e4_4N2|rMa8&dX-w#y?lfFoL(`WJlU6B?m*wO*8W9$);`|dnh6}S4(l5s{%e&7VPww`DEJ}!mt+fwgdJQk%`96!XR!)pXE)+IVrp5X|b1N11uJOg!z*$9goTdFY`0#)+id~7X58Cdm(DkwoQMC^`IYX$Cx2vST#los%i1>f$46<A+@x4`QTGo%S9;TZiDa0{Wg^~0yn7L9x~v`+E&cbJm6P6I27=W{_Jfjg0>BC#;Pv$e>;xzEx#WK><N$Vgvjd3+P62or<}u`zu9B?J0lZeoUQVQ1chpG95<<x3VYUb1z5`hzwlK(?r+JhU#+BSTsw0on;AkQR_DZv$v}n<a{11fMjSAi9B2$w8=C1rRcgT#TJ1aTu3ixh#SlFhP_R8DO#n5>Mm^-X$<#z7{~aRD40}B$2~Zgd-(1v`$Kk#TUk$c{;G|m0jan?2y1(EM0Ait_4p63l;=q9By?~cDN2~0?NEmsxn8E0P_9$szi?URSqafVIin(ab0JO!fg2{yTHsbt8ou|M)i5ekA4pFs>2*6US?TRtj$RAueCX<g`SIgQNk)0tz^{&IKv;0EYG^`EdyYnok2bx1gr8E*y`UR8q_;vKr1m&J7jP_MjD--T^>Vk=eqY9NNCkPi3)ML%2oyLAC1pIRDf85RY1^I0iq~Ns@;Ga%R<0PbYJ(jN89^@@o2EU{}$7LWUMpiCsEBk6^pp_j$q!=7PoL4uwj3KwX;|yWQSDTJ5RC<77zKcR+|)CK$a?S!vbf6Y6dL?-=T*I7O9I?Qk=p@#sCG1lR)AF8>`H6jClmeHY=C~hV5noc5R%(-n;H;IY60^b?#C1HFAlK=(8q~sWplSaGS_OL*WYF1qkw>#2gl^m<#^{WG67*hB$!!(-z*GfBFLAIhE49JMyhfNGw29qd>yM)@#nT>4{A}Sk+K8U$1-A5s<bR!X~`0d#f_UrW7Lt17nUKVB)X-Ie4`gyoxSg{Wy8`aq{ZoUq}gRbwidnAioi^Li?#r;wCvTaof;t(#ATbU%-|6jDw=kNYzRroAg23-O{qipaD{tUOs~rMSHBk$l0P{rjrQt<U7hyI)~P4YdqTCRlojZjZODD0l3(&Z~et;G@f+&Z=6t84~ZLLOfC0K&{~!vH#99j$;we2meS~}fj6yH%?a<f?NBqz?SrE|17odgW06_OFUZ{|-9fno_SzQg$j!lNDXWI~f+z47Pnj-q6>nh6W1!M!+2^1&;Zg|O<u26*VBPb#RglE<M+;M_>>^npk(O^s?4?RpRT33R3c|N2!Si8n9{H%MfY2i=t`fhV9GINE>z;a#vcP=9?MWteXX9$nyhb4E<QO0Uk>=}GqRr5lM)t_o8h(U-{+i94*RK`)tLq-TzwQ~Sn^n17mC%QF0pktr_%SxE^V{uw9(2Pcu(p8eV6_%oIPi53zQZ`fiXs~FzXEvf$VqgXX6G1`31Z~F@#+zYHWG@Ay5oC^Z#(~zn`=>m;gP6lE6S|#Xf*001jJZ9?*hZ3L}G*!R@0+UKgtULQU$$UvZ)gs3rXJs&2RjgLDd*0d611H5Q?$kr;6ti85qbY)2-ICuZj#%ns&~U)&+e@10IS;R(1db6caiVIN^>){Z6p;(pRemQ|0bgs_V3~E25i=;e3@Ol*DrK>)zF9@Fp0{r(gGXpAfGW!mQCXlwY(YSPvBjd9@}rtD&awi`9G{KWI<+0f_tSp0pv<N_8xi0uED>9~-##Yl>dgPrHd?LZ}jb1}J6uR9yK2NK|bj`)l*8;%Wt$rY|C;3S9s%f`~vPf~9pWTPBhZ@R^d4lX}n&`=Mi^llCkT9?7{0`4q!Qh$Vh-+H$aS3|@hDcYaRbwZM>a_^>^d4t0Sf`L<$@2ijdtpPp5}u2n~gEEPF=UMEZM!YALrr!qk@nw{Ayf)%DRsBCU3%Xa}rY``mx6JdJ9(hIh+KrjR=4Ath@g~c9aA@$|0z;L&wDk;tndx(b|QlM9I#*97h${Y2#v1C4uH&7c=qx!qb<y8Wv3NCm&Ck3NxX=>F=JF)88u6vOPR*JZ(sw*FfT9|7M&8d4OgE26tHWM%%Mj@(1Mwa`l^e)Z5q^&L9fTPWvbWG*@H#f3QtxkbJbh_$z+eSc*^523dc>(3T-m|hZNbg~^$CoXo9Bnf^c?J9w{8t}|YI#k3iPUku@+w7zT&a;Zl&r>$Qt?HBQ|(*e8n(4+x5m6y45#;yIFrt*SWG=dCo5h$ZN_RqQK*_w)Kdk8o>!Cds!=vm)wC#OCPCZIEKmBJa>F2jY(dZ8VmJv_cPa9c>Z3k?K6!5yOIYqWy%|31zqQe6+sk@(MPD0}7_!xs)?%s%^W1<=-5ES!pJQ<avgx0}b$O`70IF%qteUM}7z}9eEG9{-tg{KRmY8csd};ilO4T+e*5){;i-5>Qilf>iq61g=s^YCC{WKK1TKon?6g&S;L~tRysn#NP7X){6-_AwtfaAVww1x%S=GthL2eU*pG^%{+lVq)+sj6}e(lu3Ztv;#GZG3YJpRh)y*<Ef-Xyurp0FM20Bm#VYiG$1Ixs~>X`C@MFzKPxRnVG5(X5_00;7b1XQ5f2<6e)-~f7NM^$Z(@?w8Vw``n1-iH`+?&@XVy*+9D_b4+Db4N51wI$M}>NK~zRHCKP%x!Bt*g%vF+&!TxNGHsYpPuMGm#(P}eS6ti=EHC;;}4W?@)K;g~gp;#ph4(>z&EV>BLxGQKKCKiX_(Joi`$Rz@c{Ot0$xls{BmC5aJDArpwZAjA3P+i!qKun5cl`nkG-jS<gmhxZs^jiLo`0;g5F|ZZ}X!|6pQAdAe#QTO--2_07H}zhng(yq#5B%B=Z8=u*O$Qv6?Wkp7SRuj!y4{D_Q;9B_QmA+CM9I2!JvAAoI+IY)L9mqQ0H9>c+MW&Z(}tmTX%Iy*XA^yy<j}!%`HI&))evLozDj<euYiNn`g79YMJz2{jSamtN|EFLf-YF2ynr1KGzqX)E+<ch0%t6aU|&9q?J`T^@DUpzhm>(Hq{GSzUWqp>sq(7?74AgRk7$5kYE$FLCrwOT$s9+$Z>>5OLT~z-pL+F-6vNI}uMKfUFs3GqjjCcwk}Iu~9}V>Y2xBhHuM=q%$H)SvE?VO98C$~-a;l%q@(7ux&Sv{{1|0!Wo<)bVPOnTzvzpq%rCf$akA5H19i8x=XZ!0O=mo)y@Hxfw9T_oZ&bk1FfRzdL)^pjy&r(<=j475GG{#kd&nlrv*{F&$)H-z=L{kbP82h@hye}||rqUAmjRwfK*85CO2Z#=)Yez~Jb5Q}FD2aeTSX{A6l1EDl=Gx>majtg<Vpcn)Cd*2_U^b~DH$fegS31s!Wle{{aw*ctF%<&uGTD*ymDj!VvH$M;?Bc>dJ$rxbzx#0Z?k68K_q(4yot>Os`sc?NpFbY^N1rc$JUc)6*YSJIuWEur-_zcjjqX|vrR}{?ne)cQqERV=BCmGTiA1=glnPT#Dlc3I1&a;4TPCWird>Xa(aqF1H8V<n@iwWQHTz8-WqeMycx@rQPA|xmtdZ*DOhOfaWYP9W^G`~L=+8N`q-=&Ab719Lz!4J|TI~%gnDho24px_{OaKwC)k(!esq%NKQryVWv7zahmL?4CTIC9~RbQ6Kr1Ehe{m5;4*y?gMo=VHT!#3q6X1vo^b-6DKZ5bFdj6WE~+l=P#_H=%emxFWY@?D<SaFv%rOtyJ0m}r!FcvJEiL-)<br0T&Y^JTV)llacdS8S-I)v8{c-kA2=gKcs%rq=V}({HU~bu7O+7oxZx^g)i>0kjP6om8bp<$)M1Kda;T3Y>Tl;R@_D#F0)MhZy+>e{eMe7wlFPe2fV8SRUxxkIE#l?2VUTXzYKvSw^r~zP^5~Z5^}~YZGK6hfON7X$7P5Dh4OlB9`iw1A4=E9DtfUAItNeTOb9o=8CR2kj44}Ocq)5F9ipcB4RC)Av$u7{Bm9zhiqo0shd|9l@0lmzsRChf>klg9Rf<!RR!7A?qN=njb8$$@|?J51*p2J!L#6TPyz>YHN<t#=L$3*OE(+&pQ?%z{c_X`^`@=>YUj@<M*32<sJR{G5S^iGM3XyDC8FwYmWRSvL4xw~(VBtXLnfSFxCx8*iwH}L$Ln|H6}cvwZO5Yz=f_9y|LLEcUR)l1_&~;kRi_@?6Od*h7KOM}{sBx96hakG<g3)@lUdw?9%3_JFNUP2pD++nz@97r4m|lCFBSopWsCr`u3mIhnIJAw<>C{|sCD8@D2P)r4`45Oa<v%d|5jCx!!J1qPzRBc<iSl-fyC!j)kK#c)Fme2Dhi|;H$*IMTfkK7R-GkkbjE+o1CiOB<S@;V#C;?C%}w;{4N&14w|B^$hU!=|G>H<^?)Yvi@2mHHE+eIYbl3^3#b^W>)s#|ChP=9L`F_6}JjLRS?#xtR4Hn>HKtC>WZ0#X6kIrmn!=2RJYSL>H+L<jX8p@0IZFl;+9SBni@u^;w(An3V-9+hz=8Nvmos`vJbbC>qpqN&*cG2Ql=sNiqKOWU885YVvKm9=u8*x3D&$0$d{<PK_ky||pio!Z4TED}6%yPg%nWC>PnaDm?z4^XVhy+aI*I3(+g`9U^@Kci$=!lDhhcjfaX8R7tS-%_W^JpC#67d;IXQ$(662$>p1{ZbW6o1DiFms9e&-zLQCJWaowEH%5hh4{J8b_wBc@U$Y44&5=hLq@@qXl@a3rbs3Il8RSCodb!DeLTF9XKwDMk6;CSd~xQKn@Bl)FLi0DvjEVVpY=2=me!vj*P*$>jHtzMAX0pX8GJEl5GIS<+JODq1-tvQ?Yh1ZbwZp+umJg_FP!t{P^A3&&TKgglCuj(Wg%zPFSPULi=<Gr|aO-8kE=|F$}O_t*gIdhIiHz<+^v)bworzzroxiqPI;u$0$!%u{rd0=&S+ZuVjdl54k{lkhVcn7^pg6B@|@S#=KjOUd?E~!QIouh0&mcdkC$U1VNwo)w=4LH`qSeQnP(z$OWG|uV0&G!Aca+l@itkW@ohPm#k&7;*&L%F)V!rJXXPO6E&m5AT>NnpV6rXBlE=G7hl(gYqd;bK4$~aw#aP_gJm#_6D%Lq%4!qfGS1<Gr~=2Qqv&z^b?k1BsOIsRlu}js*f!N(R!~irze1@|*g|57Zn>#dO(A5sM;oy%x-?5uU#1tTsO}(S2TQ(4x_p0fdUEk&E7E9S^84b6jTtLkJ&Zhc&tZK9$M1;WnH3a99#GsP>Y=DZjQYun#`fDB&Ffw!;uJkr!xz{A21%0_6z5R7h_9Tg9Gwm~49m^yEv_|dIA&Oyue(87%PwaPnzsx$t;<WCYR*h6Z?kfYDG^NR>N-vRWM0T;y0L5A*9JgPLfpKXWr6CiVoBh6b|`CuR?-avCS=U=`Oen$meui2JcGlfpsH{VT=;jN-ye;)hhvNT>31Yxga{Uj&lFNDfsp~ZM`t``*9YrEoUR`5mb5G%$P~WC<HsxyEqRi*kfZEv?+y1y!|{$RW)nF8<g=BUE)Jq#iDxs(1e1qL3IFm-X32318JdST)vVA|y9Jph57E&9I+x>JbU_4`xmS(nIeaC9&N`-5&!7zm_-tN+DNd)wR(Xg!5oNqvUqmtBe5czu4>w3+K$b6el(n>Ie~H2<!o5!Upd3+&&Ylge${vC3M`zKw;;CBit?Ggb`8F5U3QW`$G>QJ$0nN3CFZpQYvqvb^xk(Vf7bj<@o?V}sRD>RctMT@D!0SJ#xomJhwy)E(?=Ftde?ETizdt`gARY|)Z-G?`j$`lb><!1G-QjrKcI~~r(Qpqw@7kyYKZ3CT`1zyHtmuC@IsN<(3Bz-M%iR5uJKoXH*{#y}*Of5OH+58W^7$&NVNtb*(roFd;lzNZUQb+#1>dqq(FpiP#Z&YlmgY~tzv=vsd{7gC`)Jd^c#R_&rEyG4KIMBVu$}F;9{K8Qy?(K3oeNt&!p=_}j<CAC!_+9>T6hR)D-!o2$(H18u?bv&!4;IyV=?6Zs+`B^zm>aa0Ez)7gr=i}rMlwgK)c8)9*d!pAz0X%-V8yKlN%?`mXi9yI#jS^V%)4XEK+}IDN`5qeOG)6d$x*`XbZPooRAGi=RCgL3YmmXRS=x0?eXm8WqdRt+f0%<Eq<vMogsKkxh@cTK#3Zwt6i@q<EiIe**n1Q!yUSRt-U_=_UT@uc2-9=TUP?l-TC#TU8z-PQeAb;irPpzm89xXq8XUWRn2rtw42r2upUXBlhdwV(I27FRX6T`!#8r`Q`s0FwRUUW!&BK?^YMS=({H?{Na~`Cykm*7kj^4l(ve7>o?K6R2-EBTd4rrH*&vlKFnJr5{xfxa%!|)#UhPUN;5dEJU^Ki#&}7!?p}jo>;rk~N-nI#ZWe|gcmi$!?2OZI<u-6X@t<$tQZ1wQXXgnINIUbGqtyvJ>aVC88S!q}x)x6`}W#?&X+yguk7bLi=GW5)|(sui(wC(<egVN0d(|y-GFWqtX2fLDmn5gMUVpJg36ycjkr}gZc``Qio!VeifVR|B=IXhuw=Z-M6R$E{w$8$>SKQ8H317*g%t(|T1%IR3Ww1YtBw32glX=7J<u9gWVmL1Ee3>?``sx~UGF}ljAt`uPAyP?EZP#{W5*tyazfQ3rSsRBn^b0G|qTgf%sB`SKJG>S~Fd^tFzi-h$J-4poieW7QXc`pTvX^jI3pUuzN8!v`xrRKmgQ}dq42|uk3@hHKjWw<2uyVaSd(q5tiCAxBywSw?l;G=v21OHplk#aWKF<flnZNym~P^b?%J1)qhSw3<QNU^vMOP;FZRQ|m*_j%0PN~b&&IyYE`$k9S{Y&c`D1&!)aJu|x2nnd05s-;(_jm}Jnj&^!9iJTgbgkXb!^-z5y7^IfL1O70|xGPR@ZLJimIr`+KNH#B~mo-_>z~fdgfoIso6V_j@)Pxg554uuOO-M~O;ea0;FIEfEhoI{dbdDvK0H<<&0ZPTo61im%=j6`YlSVnGy4K7m@6@luKh>HPb?m3R_be{IfTZxz6Nl|a;i4?}$(;6!rKn?1=G|vCSIDFEj>J}5`TApHRJ!*ACn#kM-mZnZPDBnx78x$-{*I=$do}I%|MRuAcu&p6W7oKYy?i?3U5Pl3^ZfAXv(Kfa7}BN^O``~|jy2lnW4|&4(Rh<an}2zJbb9eUUZVTq!`a_a{$cx;)?d%}IvLjUeA|xb=6W64dX8^HlkDCqFPGs*v*=!X(S*~*Tt+0btxaUPcv`t!qyUD`sa2&;U)qgt#I6pMy-e$S>bYL*YVd+&1H2*f=<?*dlMg4ChBhQ_XrOFRt-1G`E7@{CTy?`HdD0uJH#J55jDqexxjp>59KnsK{;JK_p1fU=lAWt1#}LK#cy}D`9PZEN2YYWuVmCV2ITYjZ{{DD&Fdv1{_M3xnv_E?@-wni`hz{n#Ubr0|Y|q6wn(uD+++M>#W4JrPXy<VIV0T-L_P6&A_YY=o_77*l&hDGpK0w)@?Tq(#x959r<_E!cG@9@3A0CFg`#~6n!JFNKLk@Zt%iv4uSAX$`LGSMDjdu=q_CVx{{e#2t&SA8@GY<}T4<iuV;ZAsPI09`w9tV2|Aq*MqAMDM?dxz0@2Vh5sNSrPajekd|j!6LmRZI&1W>B@IHviEfXlZIoK)8;YacFlCi6o|L#g@S%2E7OO<DSdGVrYJ*=PjFe_R-~Y^)+)<ytC6FM4j_iWyfT|lkbl!T#Fo8{0%Cib^8}ZJF8lLu=AKJhlnP#&+q-~OHcpvf1(l;<p'
PAYLOAD_SHA256 = '2df296371964507a8ec5b39c405f322351c3c826a9181e7dadf706d5833372b5'
SCHEMA = 'paper3-r1-resume-delivery-payload-v1'
MARKERS = {'action_startup.py': 'PAPER3_R1_RESUME_ACTION_V3',
           'r1_spawn_bootstrap.py': 'PAPER3_R1_RESUME_OWNED_STAGE_V3',
           'audited_runtime.py': 'PAPER3_R1_RESUME_RUNTIME_INSTALL_V3'}
DIGEST_MARKERS = {'action_startup.py': 'PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1',
                  'r1_spawn_bootstrap.py': 'PAPER3_R1_OWNED_DIGEST_CACHE_V1'}


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary = path.with_name('.' + path.name + '.r1-resume-' + str(os.getpid()))
    try:
        with temporary.open('wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def option(command, flag):
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def packages(runtime, home):
    """Follow R1 launch and scheduler declarations, not scientific data trees."""
    launch, state = read(runtime / 'launch.json'), read(runtime / 'scheduler-state.json')
    values = [launch.get('package')]
    if isinstance(launch.get('worker_command'), list):
        values.append(option(launch['worker_command'], '--package'))
    for slot in state.get('slots', {}).values():
        for attempt in slot.get('attempts', []):
            if isinstance(attempt.get('command'), list):
                values.append(option(attempt['command'], '--package'))
    found = set()

    def add(value):
        if not isinstance(value, str):
            return
        path = Path(value).expanduser().resolve()
        if (path / 'round1/action_startup.py').is_file():
            path = path / 'round1'
        if all((path / name).is_file() for name in ('action_startup.py', 'kernel_bridge.py', 'worker.py')):
            found.add(path)

    for value in values:
        add(value)
    if not found:
        for pattern in ('paper3-r1-repairs/*/active-round1/*/action_startup.py',
                        'paper3-parallel-migrations/*/round1/action_startup.py'):
            for path in Path(home).glob(pattern):
                add(str(path.parent))
    return sorted(found)


def bootstrap_files(runtime, found):
    paths = set()
    for package in found:
        path = package / 'r1_spawn_bootstrap.py'
        if path.is_file():
            paths.add(path.resolve())
    launch = read(runtime / 'launch.json')
    if isinstance(launch.get('direct_package'), str):
        path = Path(launch['direct_package']).expanduser() / 'r1_spawn_bootstrap.py'
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths)


def strip_block(text, marker):
    begin, end = '\n# ' + marker + '\n', '# END_' + marker + '\n'
    if begin not in text and end not in text:
        return text, None
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('Incomplete or duplicate R1 operational hook: ' + marker)
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[:start] + text[finish:], text[start:finish]


def load_payload():
    data = zlib.decompress(base64.b85decode(PAYLOAD_B85))
    if sha(data) != PAYLOAD_SHA256:
        raise ValueError('Embedded R1 resume payload SHA256 differs')
    value = json.loads(data)
    if value.get('schema_version') != SCHEMA:
        raise ValueError('Unknown R1 resume payload schema')
    helpers = value.get('helpers', {})
    if not {'r1_clock_recovery.py', 'r1_driver_compat.py'} <= set(helpers):
        raise ValueError('Both narrow R1 resume helpers are required')
    for name, source in helpers.items():
        if Path(name).name != name or not name.endswith('.py') or not isinstance(source, str):
            raise ValueError('Invalid embedded helper name or source')
        compile(source, name, 'exec')
    for key in ('startup_install', 'bootstrap_install', 'runtime_install'):
        names = value.get(key)
        if not isinstance(names, list) or any(name not in helpers for name in names):
            raise ValueError('Invalid helper installation scope: ' + key)
    return value


def validate_source(original, name, payload):
    text = original.decode('utf-8')
    preserved, _ = strip_block(text, MARKERS[name])
    canonical = preserved
    if name in DIGEST_MARKERS:
        canonical, digest_block = strip_block(canonical, DIGEST_MARKERS[name])
        if digest_block:
            # The existing cache remains byte-for-byte intact; verify that its
            # run_path target is the previously published helper, not an unknown overlay.
            tree = ast.parse(digest_block)
            helper_paths = [node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'run_path' and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)]
            expected = payload.get('existing_digest_helper_sha256')
            if (len(helper_paths) != 1 or Path(helper_paths[0]).name != 'r1_digest_cache.py'
                    or not expected or sha(Path(helper_paths[0]).read_bytes()) != expected):
                raise ValueError('Existing R1 digest helper lineage differs')
    expected = payload['lineage_sha256'].get(name)
    if sha(canonical.encode()) != expected:
        raise ValueError('Unrecognized R1 operational source: ' + name)
    return preserved


def loader_source(helper_paths, names):
    # The actual helper source digest is bound into each generated import key.
    specifications = [(str(helper_paths[name][0]), helper_paths[name][1]) for name in names]
    return (
        'def _paper3_r1_resume_load_helper(_path, _expected):\n'
        '    import importlib.util as _util, sys as _sys, hashlib as _hashlib\n'
        '    from pathlib import Path as _Path\n'
        '    _key = "_paper3_r1_resume_" + _hashlib.sha256((_path + _expected).encode()).hexdigest()\n'
        '    _module = _sys.modules.get(_key)\n'
        '    if _module is None:\n'
        '        if _hashlib.sha256(_Path(_path).read_bytes()).hexdigest() != _expected:\n'
        '            raise ValueError("Installed R1 resume helper SHA256 differs")\n'
        '        _spec = _util.spec_from_file_location(_key, _path)\n'
        '        if _spec is None or _spec.loader is None:\n'
        '            raise RuntimeError("R1 resume helper loader is unavailable")\n'
        '        _module = _util.module_from_spec(_spec)\n'
        '        _sys.modules[_key] = _module\n'
        '        try:\n'
        '            _spec.loader.exec_module(_module)\n'
        '        except BaseException:\n'
        '            _sys.modules.pop(_key, None)\n'
        '            raise\n'
        '    return _module\n'
        'def _paper3_r1_resume_install_helpers(root):\n'
        '    for _path, _expected in ' + repr(specifications) + ':\n'
        '        _paper3_r1_resume_load_helper(_path, _expected).install(root)\n')


def patch_source(original, name, helper_paths, payload):
    text = validate_source(original, name, payload)
    marker = MARKERS[name]
    begin, end = '\n# ' + marker + '\n', '# END_' + marker + '\n'
    if name == 'action_startup.py':
        block = (loader_source(helper_paths, payload['startup_install']) +
            '_paper3_r1_before_resume_active = active\n'
            '@contextmanager\n'
            'def active(root, action, *, modules=None, torch_module=None, emit=None,\n'
            '           action_file=None, traceback_interval=300):\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    with _paper3_r1_before_resume_active(root, action, modules=modules,\n'
            '            torch_module=torch_module, emit=emit, action_file=action_file,\n'
            '            traceback_interval=traceback_interval):\n'
            '        yield\n')
    elif name == 'r1_spawn_bootstrap.py':
        block = (loader_source(helper_paths, payload['bootstrap_install']) +
            '_paper3_r1_before_resume_owned_stage = bootstrap_owned_stage\n'
            'def bootstrap_owned_stage(root):\n'
            '    result = _paper3_r1_before_resume_owned_stage(root)\n'
            '    if isinstance(result, dict) and result.get("status") == "NOT_APPLICABLE_NO_PARALLEL_LAUNCH":\n'
            '        return result\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    return result\n')
    elif name == 'audited_runtime.py':
        block = (loader_source(helper_paths, payload['runtime_install']) +
            '_paper3_r1_before_resume_runtime_install = install\n'
            'def install(root, *args, **kwargs):\n'
            '    result = _paper3_r1_before_resume_runtime_install(root, *args, **kwargs)\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    return result\n')
    else:
        raise ValueError('Unsupported R1 operational hook')
    fixed = (text + begin + block + end).encode()
    compile(fixed, name, 'exec')
    return fixed


def preserved_sources(root, found, bootstraps):
    paths = set()
    for package in found:
        paths.update(p for p in package.glob('*.py') if p.name not in MARKERS)
    for bootstrap in bootstraps:
        paths.update(p for p in bootstrap.parent.glob('*.py') if p.name not in MARKERS)
    for relative in ('p3work/meta_training.py', 'p3work/source_training.py',
                     'p3work/contracts.py', 'tools/r2_backup.py'):
        path = root / relative
        if path.is_file():
            paths.add(path)
    return {path: path.read_bytes() for path in paths if path.is_file()}


def refresh_current(runtime, node_helper, *, python=None, runner=None):
    """Reuse the audited allocation-local pretraining-only refresh protocol."""
    import concurrent.futures
    import subprocess
    if runner is None:
        runner = subprocess.run
    python = python or sys.executable
    tasks = []
    for slot in range(3):
        row = read(runtime / 'worker-status' / ('slot-%d.json' % slot))
        job = str(row.get('job_id', ''))
        if job.isdigit() and row.get('child_pid'):
            tasks.append((slot, job))
    emit('R1_PRETRAIN_REFRESH_CHECK_STARTED', allocations=len(tasks),
         healthy_training_children_preserved=True, new_slurm_allocations=0)

    def one(item):
        slot, job = item
        command = ['srun', '--jobid=' + job, '--overlap', '-N1', '-n1', '-c1', '--mem=256M',
                   '--gres=none', '--time=00:02:00', python, '-I', '-S', '-B', str(node_helper),
                   '--runtime', str(runtime), '--job-id', job, '--slot', str(slot)]
        try:
            response = runner(command, capture_output=True, text=True, timeout=40)
            value = None
            for line in reversed(response.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict):
                        value = candidate
                        break
                except ValueError:
                    pass
            if response.returncode != 0 or value is None:
                return {'job_id': job, 'slot': slot, 'status': 'WAIT_NEXT_CHILD',
                        'reason': 'allocation_step_not_available', 'returncode': response.returncode}
            return dict(value, job_id=job, slot=slot)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'job_id': job, 'slot': slot, 'status': 'WAIT_NEXT_CHILD', 'reason': type(error).__name__}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one, tasks):
            results.append(result)
            emit('R1_PRETRAIN_REFRESH_RESULT', **result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--apply', action='store_true', default=True)
    parser.add_argument('--refresh-pretraining', action='store_true')
    args = parser.parse_args(argv)
    root, home = args.root.expanduser().resolve(), args.home.expanduser().resolve()
    runtime = root / 'runs/training_campaign_parallel_v1'
    payload = load_payload()
    if args.refresh_pretraining and 'node_refresh.py' not in payload['helpers']:
        raise ValueError('Embedded pretraining refresh helper is unavailable')
    emit('R1_RESUME_REPAIR_STARTED', imports_torch=False, full_data_rescan=False,
         refresh_pretraining_requested=args.refresh_pretraining)
    folder = home / 'paper3-runtime-fixes/r1-resume-v3'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'apply.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content_folder = folder / PAYLOAD_SHA256[:20]
        helper_paths = {name: (content_folder / name, sha(source.encode()))
                        for name, source in payload['helpers'].items()}
        found = packages(runtime, home)
        if not found:
            raise ValueError('Existing R1 operational packages were not found; no files were switched')
        bootstraps = bootstrap_files(runtime, found)
        prepared = []
        for package in found:
            for name in ('action_startup.py', 'audited_runtime.py'):
                path = package / name
                if path.is_symlink() or path.stat().st_uid != os.getuid():
                    raise ValueError('R1 operational target is not an owned regular file')
                original = path.read_bytes()
                prepared.append((path, original, patch_source(original, name, helper_paths, payload)))
        for path in bootstraps:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                raise ValueError('R1 owned-stage bootstrap target is not an owned regular file')
            original = path.read_bytes()
            prepared.append((path, original, patch_source(original, 'r1_spawn_bootstrap.py', helper_paths, payload)))
        # Do not import helpers or native training modules in this installer.
        science = preserved_sources(root, found, bootstraps)
        metadata = {p: p.read_bytes() for p in (runtime / 'launch.json', runtime / 'scheduler-state.json') if p.is_file()}
        for name, source in payload['helpers'].items():
            path = helper_paths[name][0]
            data = source.encode()
            if not path.is_file() or path.read_bytes() != data:
                atomic(path, data)
        changes = []
        for path, original, fixed in prepared:
            backup = folder / 'backups' / (sha(original) + '.py')
            if not backup.is_file():
                atomic(backup, original)
            if original != fixed:
                atomic(path, fixed)
            row = {'path': str(path), 'before_sha256': sha(original), 'after_sha256': sha(fixed),
                   'changed': original != fixed}
            changes.append(row)
            emit('R1_RESUME_OPERATIONAL_HOOK_INSTALLED', **row)
        for path, data in science.items():
            if path.read_bytes() != data:
                raise RuntimeError('Preserved R1 source changed concurrently: ' + str(path))
        receipt = {'schema_version': 'paper3-r1-clock-driver-resume-repair-v3',
                   'applied_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'payload_sha256': PAYLOAD_SHA256,
                   'helpers': {name: {'path': str(path), 'sha256': digest}
                               for name, (path, digest) in helper_paths.items()},
                   'operational_files': changes, 'owned_stage_bootstrap_files': len(bootstraps),
                   'preserved_native_sources': [{'path': str(path), 'sha256': sha(data)} for path, data in sorted(science.items())],
                   'original_metadata_sha256': {str(path): sha(data) for path, data in metadata.items()},
                   'scientific_files_changed': 0, 'checkpoint_files_changed': 0, 'plan_files_changed': 0,
                   'jobs_submitted': 0, 'jobs_cancelled': 0, 'signals_sent': 0, 'other_rounds_changed': 0,
                   'activation': 'FUTURE_R1_ACTION_AND_OWNED_CHILDREN_CURRENT_TRAINING_PRESERVED',
                   'refresh_pretraining_requested': args.refresh_pretraining}
        write_json(folder / 'APPLIED.json', receipt)
        write_json(runtime / 'resume-repair.json', receipt)
        emit('R1_RESUME_REPAIR_APPLIED', payload_sha256=PAYLOAD_SHA256, packages=len(found),
             owned_stage_bootstrap_files=len(bootstraps), checkpoint_files_changed=0,
             jobs_cancelled=0, jobs_submitted=0, activation=receipt['activation'],
             record=str(folder / 'APPLIED.json'))
    if args.refresh_pretraining:
        receipt['refresh_results'] = refresh_current(runtime, helper_paths['node_refresh.py'][0])
        receipt['signals_sent'] = sum(row.get('status') == 'PRETRAIN_CHILD_REFRESH_REQUESTED'
                                      for row in receipt['refresh_results'])
        write_json(folder / 'APPLIED.json', receipt)
        write_json(runtime / 'resume-repair.json', receipt)
    emit('R1_RESUME_REPAIR_COMPLETE', jobs_cancelled=0, jobs_submitted=0,
         signals_sent=receipt['signals_sent'], record=str(folder / 'APPLIED.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
