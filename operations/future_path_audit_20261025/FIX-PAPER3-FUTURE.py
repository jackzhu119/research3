#!/usr/bin/env python3
"""Install the owned-child correction and checkpoint-safe recovery command."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zlib

PAYLOAD = 'c-qBWNq4JAll?EX?$Za5^{VKdg{=V%wkAMqz9YaAk`SVvM)aNj?|maOlR>s$z3xpfd@KvdAu@&=Hv<3rkDvO#j@j~>{wp{fPRG;FW%i$0{>Oj)(cAPk^Su*?-PZAA+R~Pt`Fi1ww58`Ky>{=`>zG&Hv*W0fT!MZ3*z4MQkWR9|ov)Yf(>mBYji9?*hmBSo8O_AES{M0yyX+)KziTJK=BaNbt!C7TFTtjD4?FD!eqY1AbH~_*?+w?!+ccJ*jXCrrNXJ-1bJy!6r?6w@k=42`yvZTVuj@tXXuVGJg70{(7%RuKn%x-omu~YOb=t?E<Lr9fNsck@{@%$4#^@4wtvK8#_rQ2s`>AtYG_VJwc^_<APH4>6SX<KTj$+!^BDFKW6D7S)`;KuA{<O7?Je%jS_B6h^#d`4k6XX$Pw6Dk;V-FL2PR@SbI@9%NkFgAMhvzThhUC#c`vzaXFN`E1Ir{r#i+%2B>?NLoEY23rq_@s8toag>EW8osEo89`8m(sK?*pzMf=;rjd?%ls9G7nI9MM{z%kh>e<WU(vfb9GfvY!e$(^wVRkLkP0*!K@_*tgc%XzbX7iPKxZPoGnq1N5YKA9%gnz%?7ti`@Y0Ubs;*a4~jca*w)p99p_DoQCu~_Pkho#^UNV#C@Jo$f4Qq(lcYBZ<{N0ZNl=3y1OIJWgB&!IIwm<tj+VfXFWq=_P}?Yt30pdqm$g_S>W6?oWm_%y4pIR@7#9k&yK9;kxM!~!ui=-@tm`b#M!Qw(96(pAQRI6`I&4*%-1?BvA3XMXA1+*zi8_aIyKmr>v!`gkLa_jC$_HjvBouIy6<)3ENo0J(8cY-y{_M9TCt9|^<mBT(GTnfd(2@c&_(Bry=Wg}aeWE7RpbGka<slxnJ?e1d*6>6W9_hQKMuN1GJ#$Pjo!i5l?-&Up%k)+VIw-Nk@kl9Po&3-bd-dtFZ@BF)6Ncasy!+_Dc0jPN?Vlj+XkI>7EGVbUK@U+vp$4-6E?kD5Ax@O-5?(`yo83Shx-xq!g-1ygV4a7iMb0J(HSy?J+@DSJY4tlij16Nit{=_=J@_O9W?AjjT7dkoVWQmzB?CdhOGT`2t95EA9=(!VBZgGnxo$t9T#qL>QAq0yD>T}W<7nh34h=m<IsrNzVJI9Zs&%&caYl2!kw%ZX`;cN5B=%xyzDqnekVzN=nb5q=}ocbjgmt%3w3k9bZ<!?GVtBo9kLEQdESNm9a9VUZOrE<;ihRU+#B}Y*6QEKd0?2@;6X?6w>r<Gv>E=fjS2Z{vQ4a6`%%gqc4@Tpz%hkBI41VmOvp}c_+j|&eOQs7)YaV*I@d4#S#wa)gLijy2}xf&P7-$b^I^$$YUX&3>^Za6hd!?Fdu=ldQt}J8qgcx=gHMTlW3F_iYWuCgb<Pxfg8i*2eZUF(zxyu5lF12cC%em)U!H8A*EkP~McTqxSM7;>_KD&Oo`;S<kvEpQ?L)S(^_Nita{XD+^?T58@`d57d@t6p_xsW{&t6@&@Z*Sc9r#A<3-XJD!k+aj#?#8!>a56D;=Vn6YQ=X%3C==`y1hef9MVnQgx<E2a64N0+tvzug{+)a;qM%i^@7fBwl~lIbjQAjbzJz`>>K|xhyEqfZrL`--_5a(r6qK&KYhORtmryyMDT0J-?Q7a)wHitzxVA1`%oUUN!NFtVFNEXci2PKcYd}x$SKMn?MwK-Z|r6^)1B#9a{{~Ze2)z1UT*%*`JJJ!sW&_H3mHMTVee$CF7#CQ=S~jY%;AgQ;ono}9CTXg#{5(r8!?Atp3tFNLi$T*%hx*aGj215?xuaIBeVve{3S3+WHl)!Vcr<}9A@%tU~@ifBoSjkp0Jmv$coQ_F~K}OVG#K_ITJqe-T8U3H+LVx7iBmnI-fYe+4X_3{LcI+=j*_Rx9szGlf>WbBtGMN&cKByS{vg4tdlXBWA2Amu{Ij>fH*@wHVzx(v)^@2p%wFIx*r%!`2DTk3itXna-XD~-w12Pz+CB|#GV;^w^+M#yn0i@2hA<w*$(r@5%#upZJqSzeNJ%>`;l@CT(b8Vv4CRF3G$jGA69EmYr_~j-&-GpJ!7^Ex(}b=oCN-zAIL9}9}%&yxq&R3YTWt{%d@&qmiqv52M*CO&I)^iP9_^*jU>G5odx^+ox}r8#8or*D)Ane_T2C2SjXsqn3DCZSF`u>`lIU0q2n20SqB(3a-DVvOc!p}=?pP|a3{xdX`8**#T93p2UET9&%MqhV;qk;a1CpXE4Z~9<Dmol<h&F(N$OCO{57z__)y4Lr?WZU2G124)%16rR{bt?W2JE9{|Og<s~^R>YxeBvCdVuM4!-lBmGM)L@e1_1UBtvs*bFwFEb@3882Xjtx!CW1>BXx6zJhGb*Ue*Kj0A>ChlJm?!A`$GSB|i0)~k2tRI_Jqdec6zKFA`4e&)~t)77<%aEb8YjH9xF^_+?|g!tU5``pA_!AEYRwz*PzTaP2nZ{)9Hud@<12|2)jDXgK!!WbGb<hj?LK&R#vj9<?oJ)_v`6&Up==Li;ZCg@;a`B$+Y(uv}pp7qxneu;B5#&#4#3Y{G3(09%q@eJiC=fE|05zZjY;Y(pm!cSG>N%;(6eaKboy*9DNx$tiin=TsU*Ky8PbLAO06WFf@{i6IeFm49)ID@}6<(g~Xe-l&7d{X#F;1Fj^K3ib{%E1Q6b(iiH=hsRWz=x3iNzGr$LFK-#GJaO}*BxDbcgTAaF`x5Lo@=nbU4V-a58?Oo)*Eu+<m)`4z$Aq~8UKnO=%};haq%o;;drsXC2yPOv}YaH^>a}7Q*4uXs&mU0n<o5savbdCbw}>M!FxKeObYxtYT}Gfup1q+xw{7Gh!b%PlsZtAbGi2>=X(`>lw-_Q+=87=;4h!baTGr7RC3z3DL4=17pBrK3SDNOMS97(w8C4IC%_*cfkzL84e=S0eTip~52b`-kh>6uA%8&r1G%FPyMQl9OwLaO!Vql}xwUy*cn*B``pkL$CX{#uI$+!o-+F8Kd&uFI5tmSmgZu}2d&1{4;REaua&EHU#ToA<rpDN;Gt(jAYd6ZJ4xXjFDDItU-4e@H^O}cPC4EKrn3AuBUEC?>vNgEAw%MAuTf^ym)Ee(@Z4cjhdaE_xwTH7;XXt6SmNRd@Os&Y2^Qt|Qj5)WHXMFFMHW}l5G{Rtg$}<$6RlcRfv~sK%_z*dhaiW|9zQ1noB_2-T%iFup&#zTpUfcsc0}e{|gi$F+C(IEvhUbF&N{;FLwLP9V@W0yVeU?n@S>n7q&S*8?Ipgs>nR=Sun(dO##2L+z<9?|_@bU01BkjvQ8Hn+Fhhjjn_wU9D4F_?oxnryZorpf~jr8t<{8GfI#wbDV1k5u`;m<FyN%&SD^U--Acd^!${l(+4ZhGVSZRqLP=iFJ%HRolfHDBj>c<jB*^S4>=I>cG^D>z8SI(aUc<N^t+vCn6{d#Pdc@_VbqC`0S|X{-Xv<*hmL3BvVph*-e59lA>RDS+RB9rAwp?;^%rMf^Z~g5NTjOX||S$UoSIq>o{KkJq;Cz1pznF>{Z>w50{P<X?+;)1q@kPBkGum3#~AvVA09-7D9<=Jj-(oNRRAKCS5|5Vl^}KIOCv@*T*BD?F)q#v2nZZc<E#t-^=fG(LRah+`vhVUi!ky*90dZ8~gF?(UohGH&?5B|RCRi5n*Uh?o;~c3z{vW+`EmCFKa+2=WCMuI1ZdJ!ZnU!e>iAgR$GD^3ex8vt1hGZ?6wA#RmSLq`tswzxBLroN!i6@}ZlMcs$s=(g(!{5B6=!feiA2mHo&(h2w7%aXI$!S=pdh`VdonM_kM|c4L+{r}k3gJ{Thx=O?ge!ON()Qp7HIlq|g#`+<)<x-)m+4O5E4^KE-I)1$Y^e6kvAu|0d8tmcXH=Da4OiDOzb&79Bm$#m?r#*_J&&n((QFDaKN{_9nTKR&hRn`_0^5UW@gxL(DF%$=3K^bNL397ZXBVE6C*q~hVA?;kGtbi~bSK4C@jynsz5CqU0YzwS$cBk=n*AI%nrimh{8T^hh*dk1(_=}YtaF%7eS=I?e799j(vsQGB01vfuP6?Yg53;+y{9Exzwsz3e2d1Qlq3~Vw%-sT{GSU>JLj)<p6UiYzwZ5gk(;Y+K+8QH69ErjHOT<8*5G1fof{J8&snUsJ2nllL><`-v8x=pq<)XjfbPmkh(%5fj}0zc&^u$wD$AZ?R#C-|9+;<%?c3E&%_^onqoaXnD{;T(Dse~I-=Zm0P;eiCthENl;PxOSFuO=TWLd*qrA=}C;wlLq-fuU=F7)&5Fu>1$rD;5mu!fzH%y`e7_tS0Y_k99zj1;aRx`@@Jz1?Efe5G{%Q6W9<RXm}U4XIFU!468!LO<u_V7_MLOAqZ*YrLf+?G1K4YA9#2!b8tkX3^*b5KLU114;Q>AevZgpj{s1;*I9b@809TFZ?DQ~?BKVcR`)P5_957#IYaaYD&ee2~)3bl*_{Rvg^zLwO`3avBrt#h68a*HS7R5{A)*CkaTOYPe*MkOhIZX=L^^3YsjZ2=YF)5C!d~D&`l+#qk98*k%zi3@B7su_W8(n(v6U4hi9x3Ht3ojzB4SE7ubcfX%(g}L>lkhn4W-3k>_(<>FJk9iGJLlEYNAvMKYJ0lvFeU_68t+Wt<k1|Mc<kwmo7Z_tRc(cIiF`isMd2QCZS(xIXN4k0i#lIEbWBl)P0S7XXu*wFxaeQyH_q*>xZA2fkg@ll^CEh_$lVIraQ&s`S99JE=Y74dvgcM&x1>3UkAq%AzuSZ%;4hz;XD|P5hGh;^U8nLZ*F)vmqs+UqQl7Q76#7<qhInFysV@cZTXIBPE5JD)Xw16J)y_om=aqVZ)KA9*mLwi(L>OjO=WO7E@;cA+&DyJLQ)?obKXOFs%;Y^b$xr&oR-|-Hae}00Zc*Q4j#+Z5*Mi4nzL|J3S7+=DJ%Ih&6pQZtI-kxpG1gz}Sbtv0qISRdA?!r8UgDe{#!}onauySKZG2E;_NUJ=?Cw^>n!o1s9pcl8KOj8Y<C;~JG3Oz8kXp@~_uI9b!vBtc&{_9$fqNv+oYlDF!WS{OCO-ogc*t#C;^P$`dnMgF1{Q3(^4(BjU#osTz^qrS`RrTkgRS^H)-&=8A>po?ZmIahwg!A;=Sd*e8%G5mX1tydU!vqo<5W2b&RYmCeh5q@=7H?&&o#qxjI!?0|KmUY`1x@>??3+aaoZeEn{+MvRq|KJW|R*M<W3tC>qbu4P2hjc^YD6*{(GcT4uUmP%@1c3QXb6l9CKU{58;OpZ$0~7?cYGI$GIB5=V!3TGo5MaD}XOj<ZtV2PW~QtdIIiYZVP!2=5%BovqiD;Oy@&++XDE*uW%HEA9&2JqlvuHF~}BS&zs0ij*<1WcBoI_njwbYXx(%Vt&_MHwpY$Gy3AST%-fUy=}_%+q%Zb*27gNV1?F>(@DH<qWIb*D9RM8<HZ7a{t=|ELatSAl`)otx`)H5cn`3SvN0JxqTlK+QDs`<YZduo!Pl(?(uR^wKItRiQ$B%I)dUlMqLC-)&I@N{vO!j`}H&mYX#61~b^m5#(t|0Pg;!fbV@)qSw6mRvCi+b#zu~S)N>X-4&-$D<L|DSy{@E^n>^eg^eU!=fY#t6Cfu=;%yx=_ZwPR#iNVT{UnT*HBGzB|rjjGx)NchuiD(Vlc2veLu$U-|H*XAEc0`KWE?BiQ9=dq|PDQ%_SP@>;9m7T30~*WnhJX}frcSI<x1j^~*9n-w14X5Wnm%(~sBt8szPlE6)X8KL{Ixq)@~8~G4$dL<@&gDbg)VZ{PlzUUHTUjOcs4!JkVowLCsUR1m-*M7za$(8?j#F2tCZE_wO<}z1PwH28c!v8-}PSzh+u%^)z+@h>cQQdGsy=P@j%9M5Y2ds(tx<34Yd8R%(=2>bOlKemHp>vC=2D=n-m3)#$aUXv6C^?xr&p|ylRC8ooQ1uza_7}a*L32!NlWgyVQ8*WsYbtB+rXF=$I-V81C8D?KZs%s9Ypy~5&xY-*oS@vd>J?CRM$AFn81fOw95Dv-)N3hRyD0k%Wj_pj^yCUZeX&dK<sxjsSuOYGCd?(yQ?eqC6MKF4%KL3Y$x-$INgS(u5$<LEP=9+3arTPuCBH7eGr!||=^i`3Ro2UAsQCEoOBqui!(FeAuQ=<?2XXHCI(rl2m=8Ky=>d4&_S%!f(i;{%EYw@cc}aJ4VqDdkB-9HRinvzoJ3Oks%A(Fdy>qD@BM&QjIuzF@YSKbRiaQ93I`a+r^DPDrD7cXRRMZ8S7Zf$N<5wIF^@2b?2iOw)TLeEiCjRBO>fG~BafRb=HD<}UOnsNg1-Hj)Ty0R^0J$`9zeas4m7cbJ(Wl1yI!c}^iXP4~e)mM(hPZvHA3Ud<M@k<))e?F$U0wUFeNHO4hS#%cZD1piX+&`vIp{0#k9+F3WA0hlI(~pRfH^8P(@WqI*EYGoyM<jy{zPKLg!z~g`5fqUP}b5bb-%5C7IhWnFBkQC!j$X%%q25N++DC9kd6i}`JQ3_a}Pxg$JcvPx{D)of>ZJ#V<l$;^KcFMeWs;#zaV)%p?6uOR#owtPmgs{*4KV@pPRo@teN<D!u=Y~n0tcy(_huO{y|qC>Io0y^O|rjaYQ!RX>Wo~$xGL2XnW*0`*HfVmwtG2Ex1_8Pp8HC-!r9e$BHja-_qjqSDaYYH{o9LMcgj{*<&u&L(Ij$F_*MK%%%L}S7TP?kF{52Eyqlh|D%1h8v`p+wJgQY;<>uN|HR*VTG5L@b@Qce60Y8<xUO=*$Uq)pwdygS*5p1NY~z)D80FfOqwH`FwR5Ii?`<zI<zVkqf5*>$hd8$h@}%2-0elcx&EZTN4Lr?RtR(&%>~8O4?X;ThdRcWm*c0$~OY6PbJQiVr>D(0hv&{;;!u;5yK1hDAhIs_PwxXU<vPp4nx5Bnd%r(hB9(vqV9YMuLiJksLezbdcxh`1T13!0Ny6ofrBpitO^LI!1PT&R3E8u@FlGD$8YxkJ5?`7?Z?VY*rl<FuB?tL}6UWz!2JZVR^_G!;m8A?3)K8@9w7r{M_(}HtvOn^nkjC(@F*CqMBe3uShUNiUJjxaa$KGFJLtaIri59yAAVOvx6uBDw{Qx`bcZmiS(tURX)?4R<FN54j#gj%P2e{Di;G0VBemD<?<BwHA#yRB=#+%v`He!DL3L4&w{lenC-?_Mv&>-!z^G!S)*6`j`?@*oZ`3oVXM)$?V(kb1JHZVaEH@_4}sk88bNx!)L?Zev0H1dK<4_nA)#?^;@L%J9vSJ2IH-u`1_fj81-s>R<Epi#o|!o!Md&h172-u%jP+*=s=c5X1@Qbwqz3agTR3cjf!SE^sD4WUscmUg8)lJWY{9*||QWb05s9&Jms>?5AJ$g9$sf;`g~G_CBch-XEui(^Q|GZgN$e)8Li+FrMVKuVk(A51hAoTDqf^z=pB#gIzJtLr;~$t$a^^cG7&2X9DZ(Oj##y023ZO?xRt@Ld4d79-gC)ejQB1pQ1nB6tyF^W#mK(AIRLF>cf<uLWi4$FKvXs^raHt?jo$KFZVDgeJS&cup4u@+5O2pSLc=t(kRE85r3nQYsiMWzq!Zr_a)Z$X3$#w`v+o0>6@Dzw}bs(hkl7|?u=g$e-_64P}_ry3uPZrfuUA}ZTg+zaY;RX@U4Wg8DG1w{kzURR>fHq`P}UizUW)nm5P(_oj3Zs78|_^_r{p%MGsVrC7=62d;{G#B|Y4+9#S6#bkkzp^yX=`pOn}5qM5c)<->YujSnsM1{`$P)^3Aqx?k5XOAMUR+ylzFHah3UA@c#mC1M>@UG|~hm<wI#RV;n|1rB@`&o+q543F~N4ERU;5)n3~9u_sP>N5~E=<UP1IGK`zPw!7`S`=G%gc$>1Fz66`L0HqrLgv=<PpVN!y}u$4K4E_uy}uxP5g~8wq_9|ZezES~+~fE1q3aKMeoYp|-o0s}xO&7E>UBJEuZfIPqSti;{V>`&uoPi}RN-kkC-1?VY35rU4-~$!XC0}Bn%5aLo+&R!zUCaMhD&xywe<pf)7iMp?@|m(2Vy-+UW8L?X9^!n7~%`eSgmVlr!uE0<yCyAbi%FoDAr_KyxRo%I;xK0P^|v(9d>uz=Febn(mRY*Zj$TPWaIEnX|-o*0lCONVfeD4<ZpQY@%uEY+sHk8knfSsyzWc#LRN@}qCc79L9zZypQ#$F#DW4(lz!^^_uyfi_qoW=!w&oF2i~goK2KfeW~X94de`T-{9>gZpfIZF;VS#m2YJjkir90ayqW4q1Fzso3tpv?*D+@Hj2OBPyjq;`Uk9-U$dvj6jU#kk^x+lt<F*+G+`m;~TAA}4OFvN0@en^V;y)%%Y|p8eZ5OS^nmL-kns1YNbMEPO+i^zi3F)}7E3;-})0^-?F6%`0U7w^_d-y}18{sG7&vwumen)3j^pJ{klfT9O8qOx{+Ss#mhfR(t?+j{r&-1r(@^j{_$nTt~r)vSfu}B@}7bq`zpFL;%uB?iw5B`GR1AZZlfjKvEp(ErUeb~;sSL40UlUK)`CC+eW!A_}X-y54noabkNkG8tP4!`sfltYB+WJT{OQ=e7XpgPx+>?=V2gIG_!#ngKxdkr4+acq9boaPO&7Fb~iU!RfuKc9!n!E15p$T3#LQM#r+9~HKz{shWN-h>^=GgxP<^4*{2V;}i1@&l7ObH&A2;f2qg-UkrzTbzg0q&;N=*%xMue)ejQ>^UrQpD;u0zV)5s&6}#;;dLhSl`nI}hde=I&UHGVz7k`2q29rP>bbMboX^tGF4U(48-#qrPkj4;Yu?GV<fWY_$l)46cW3%Y`sZ7%Zu=QS(|b9RZ^D|Oi`haqnG5be`XK98?5Du<|7h!@`Pi}N+%wk|Iw1XIp$`Ik>1^`@!p*>cl{2a9bA`(;-j{dBN`Dn6!@Zhe>L(bBeZgmrsL<zv3n6>=>iF-l$ZPe97v_v*?GL`RUgx7)wC7)zuH+)icfX2yh}{^Ld@QfQ<X(e_-XQ?qs?{e*CsM(EQ~#Pzaq8u{#0Ay&NH&CrcD1}l#Qu^`gRC!XdkvdvWpRPi)O&~Y4#8djT;sEv{GNbk(EFac!nx%;JHP;y-nSA1=;j07y@?#Fc=t^7sJ*-Y>-QY~6u$-MPuvK<<D-|otZMAy-02<j7hPL&y^GK1JfS|gbX@j{GZ#C`eCny$!2k2RMeXAt8QBwk(w^_MW9>UFd2=55O?5pTQLhc~XnbF~A@<)sE%uYRXJxCml-@&rK8IW%*?s$N#b95n&C@$Bm(Ys;T;4N7=IB2MiaX8;cOx&$*V!ZgvCOMEdOeRCC>~RC2$Yi)wVvX;#H2WbHJdC)_?v=1blZ>>^#MkP#7PBzC~C}3{_Ya*LA8@nGT6~)hx-cV@_N5BNkU_klV2e{eRt@+EOYDWe^uTqu&Ex0oO`mP_psF%eUalZJ|6IUcvw4P6mdO~YeR>Jmu1I9{4jqkyUvfLm*@-kiDST$dIY-gy|Ndg?8Ojk4L8l$?@%nVQ^HuJEA@P4(hNIujxWHB$TyQMy@xi_o%y_tpB(<uYwJs@M@m0LYYZD$pQ;a*IPIVISm4UayD==k!5%gMoQ5;dxu=uj%4dxHd-J%?6MQ^u5cy)KE#7h6YN_}i4g4OqPW;qwulZBXdyB>Y8hJ7)R>K$Z|3io>`{MqDyIzbf_sQoF=0ewsd+&N#YZm<4S-vOBT-KoA|K;B$r(ea#<+Y^lvnm`D;qO)5-^bfu#liXOtu;%`>BJd1{65FdX-(R@N*omHEo<YVo>}I7)tbWRT$$uT?^?3WP)sdqxlvT~FIVoNe0<Sp>5%OMzanc~#^_4@@Q?FAr%qhUgl=<8QMC+RQ#D?kxlVOOs!L>pvakHR`^$H29{Y`}y+zzNMHrmwc4Jd=#`uA6Wxe(ZtH_uSzn(f7*}l9ldwd5==ucDqPm`oP+vlsimwCRP^@$@<xjg)cUdeHroL3Q66j*+pE||Oh7W=EZ$DZphd0b$2e?P)LujkpT`7+hbwp-`d>3>38KjoLxzf$Ph``ijY_4(2L)XPA96I?6*+}Ef6A4c=5ab=#Mhn8O4XWvi7XT;~9^!`R)_GWm7`Lo`OTs_k|e<)(etsE;~%YEp%=v8xe1)tij<Nyw^QqG)e_Q*#8KhS$)T6otEj~pIiT)|6Ge3Sm3zK7i-hXD?t_s^J{5IRD=nH$_=zR%SEA9?<Vack!#-k;|@sNl~f7c#k1{I7EkpSV%!yB_$AM-2P&OmUdbdnLGOXXV{B)7H%u$K}LaAs1Hto3js(`n2`-2jwswJq8|Nj`xrM{vX>JrsD'
PAYLOAD_SHA256 = '509be6aa4cfeee1d35a59af7a6bbf6b0562fc394de9c55e4e24af72352e2df56'


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temp=path.with_name('.'+path.name+'.future-fix-'+str(os.getpid()))
    try:
        with temp.open('wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.chmod(temp,mode);os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def put(path,value):atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def apply(root,home):
    raw=zlib.decompress(base64.b85decode(PAYLOAD))
    if sha(raw)!=PAYLOAD_SHA256:raise ValueError('Embedded future repair differs')
    files={name:base64.b64decode(value) for name,value in json.loads(raw).items()}
    folder=home/'paper3-runtime-fixes'/'future-path-v2'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,data in files.items():
            if name not in ('r1_spawn_bootstrap.py','resume_existing.py'):
                raise ValueError('Unexpected embedded operational file')
            compile(data,name,'exec')
            target=folder/name
            if not target.exists() or target.read_bytes()!=data:atomic(target,data)
        helper=runpy.run_path(str(folder/'r1_spawn_bootstrap.py'),run_name='_paper3_future_bootstrap')
        launch_path=root/'runs/training_campaign_parallel_v1/launch.json'
        r1={'status':'NO_EXISTING_R1_PARALLEL_LAUNCH','scientific_files_changed':0}
        if launch_path.is_file():
            launch_raw=launch_path.read_bytes();launch=json.loads(launch_raw)
            package=Path(launch['package']).expanduser().resolve()
            direct=Path(launch['direct_package']).expanduser().resolve()
            target=direct/'campaign_recovery.py'
            original=target.read_bytes()
            fixed,changed=helper['corrected_recovery_source'](original.decode())
            fixed=fixed.encode();compile(fixed,str(target),'exec')
            # Both directories occur in the inherited owned-child sys.path.
            for destination in {package,direct}:
                path=destination/'r1_spawn_bootstrap.py'
                if not path.exists() or path.read_bytes()!=files['r1_spawn_bootstrap.py']:
                    atomic(path,files['r1_spawn_bootstrap.py'])
            backup=folder/'backups'/('campaign_recovery.'+sha(original)+'.py')
            if not backup.exists():atomic(backup,original)
            if original!=fixed:atomic(target,fixed)
            if launch_path.read_bytes()!=launch_raw:
                raise RuntimeError('R1 launch changed concurrently; scientific files were not written')
            r1={'status':'OWNED_CHILD_BOOTSTRAP_INSTALLED','package':str(package),'direct_package':str(direct),
                'path':str(target),'before_sha256':sha(original),'after_sha256':sha(fixed),
                'changed_entrypoints':changed,'scientific_files_changed':0,
                'launch_preserved':True,'bootstrap_sha256':sha(files['r1_spawn_bootstrap.py'])}
            emit('R1_FUTURE_CHILD_FIX_APPLIED',**r1)
        record={'schema_version':'paper3-future-path-repair-v2',
                'payload_sha256':PAYLOAD_SHA256,'r1':r1,
                'resume_helper':str(folder/'resume_existing.py'),
                'recovery_rounds':[5,6,8,9],
                'scientific_files_changed':0,'plans_changed':0,'checkpoints_changed':0,
                'slurm_jobs_cancelled':0,'slurm_jobs_submitted':0,
                'existing_r45689_runtime_corrections_replaced':False,
                'note':'Default installation does not restart any current job, controller or uploader. Owned children load corrected module on their next start.'}
        put(folder/'APPLIED.json',record)
        runtime=root/'runs/training_campaign_parallel_v1'
        if runtime.is_dir():put(runtime/'future-child-fix.json',record)
    emit('FUTURE_PATH_REPAIR_INSTALLED',record=str(folder/'APPLIED.json'),
         owned_child_bootstrap=r1['status'],resume_helper=str(folder/'resume_existing.py'),
         checkpoints_changed=0,jobs_cancelled=0,jobs_submitted=0)
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--resume-round',type=int,choices=(5,6,8,9),action='append',default=[])
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    emit('FUTURE_PATH_REPAIR_STARTED',imports_torch=False,data_download=False,full_test=False)
    folder=apply(root,home)
    if args.resume_round:
        # Explicit recovery calls the original installed controllers with their
        # saved plans. Default installation never reaches this branch.
        helper=runpy.run_path(str(folder/'resume_existing.py'),run_name='_paper3_saved_plan_resume')
        for number in dict.fromkeys(args.resume_round):
            helper['main'](['--root',str(root),'--round',str(number),'--execute'])
    return 0


if __name__=='__main__':raise SystemExit(main())
