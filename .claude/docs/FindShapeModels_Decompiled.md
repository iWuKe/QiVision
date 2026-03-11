# find_shape_models 反编译分析

> 多模型同时搜索 (plural)。与 find_shape_model (单数) 共享核心子函数，
> 关键差异在于：共享金字塔、逐模型 level 参数、跨模型候选并行、结果含 modelID。
> 基于用户提供的反编译代码片段分析，非完整 IDA 输出。

---

## IDA 反编译原始伪代码

<details>
<summary>find_shape_models 完整反编译 (点击展开)</summary>

```c
// Hidden C++ exception states: #wind=57
__int64 __fastcall find_shape_models(
        void *a1,
        int a2,
        int a3,
        __int64 a4,
        int a5,
        int a6,
        int a7,
        int a8,
        int a9,
        int a10,
        int a11,
        __int64 a12,
        int a13,
        __int64 *a14,
        int *a15)
{
  __int64 v25; // r12
  int v26; // r13d
  int v27; // r15d
  __int64 v28; // rdi
  __time64_t v29; // rbx
  __time64_t v30; // rsi
  __int64 v33; // rdi
  __int64 v34; // rcx
  __time64_t v37; // rdi
  int v38; // eax
  __int64 v39; // rcx
  int v40; // edx
  int v41; // eax
  int v42; // ecx
  __int64 v43; // rax
  int v44; // r11d
  __int64 v45; // r9
  __int64 v46; // r12
  __time64_t v47; // r15
  __time64_t v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // rcx
  int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // r8
  int v54; // eax
  __int64 v55; // r10
  __int64 v56; // rdi
  int v57; // r8d
  int v58; // r8d
  int v59; // eax
  int i; // edi
  _QWORD *v62; // rax
  __int64 result; // rax
  int v66; // eax
  __int64 v68; // rdx
  __int64 v76; // rdx
  __int64 v79; // rbx
  size_t v81; // r15
  _DWORD *v82; // rdi
  _DWORD *v83; // rsi
  __int64 v84; // rdx
  unsigned __int8 v85; // cl
  int v86; // eax
  int v87; // eax
  __int64 j; // rcx
  __int64 v90; // rdi
  size_t v91; // rsi
  char *v92; // r9
  char *v93; // rbx
  __int64 v94; // rdx
  char v95; // cl
  __int64 v97; // rbx
  size_t v98; // rdi
  _QWORD *v99; // rax
  __int64 v102; // rbx
  size_t v103; // rdi
  _QWORD *v104; // rax
  __int64 v106; // rbx
  size_t v107; // rdi
  _QWORD *v108; // rax
  int v109; // r8d
  int v111; // edx
  __int64 v115; // rcx
  __int64 v116; // r9
  int v117; // edi
  __int64 v119; // rcx
  int v120; // r12d
  int v121; // r13d
  _QWORD *v122; // rdi
  int *v123; // r15
  int v136; // r8d
  _QWORD *v137; // rsi
  __int64 v159; // rdx
  __int64 v160; // rdi
  __int64 v161; // rax
  unsigned __int64 v162; // rbx
  _QWORD *v163; // rax
  __int64 v164; // rdi
  __int64 v165; // rdx
  int v166; // ebx
  __int64 v167; // r10
  __int64 v168; // r9
  __int64 v169; // rdi
  char *v170; // r8
  int v171; // r10d
  __int64 v172; // r11
  __int64 *v173; // rax
  _QWORD *v174; // rax
  __int64 v175; // r8
  unsigned __int64 v177; // r8
  unsigned __int64 *v179; // rax
  int v182; // edi
  int v183; // edx
  int v184; // edi
  int v186; // ecx
  unsigned __int64 v187; // rsi
  int *v188; // r9
  char *v189; // rbx
  __int64 v190; // rdx
  unsigned __int8 v191; // cl
  int v192; // eax
  int v193; // eax
  int *v194; // rdi
  unsigned __int64 k; // rcx
  int m; // r10d
  int v197; // r9d
  int v198; // ecx
  unsigned __int64 v200; // rdi
  char *v201; // r9
  char *v202; // rbx
  __int64 v203; // rdx
  char v204; // cl
  __int64 v205; // r9
  __time64_t v206; // r10
  __time64_t n; // r14
  __int64 v208; // rdi
  int ii; // edx
  int v210; // ecx
  __int64 v212; // rbx
  _QWORD *v213; // rax
  int v215; // r13d
  __int64 v216; // rbx
  _QWORD *v217; // rax
  int v218; // esi
  int v219; // r8d
  __int64 v228; // rdi
  __int64 v229; // r11
  __int64 v230; // rdx
  unsigned __int64 v231; // r10
  int v236; // edi
  __time64_t v237; // rbx
  __int64 v238; // r10
  _QWORD *v239; // r9
  __int64 v240; // r8
  void **v241; // rax
  __int64 v243; // rcx
  __int64 v244; // r12
  __int64 v245; // r13
  unsigned __int64 v246; // r15
  __int64 v247; // rsi
  __int64 v248; // r14
  unsigned __int64 v249; // rdx
  __int64 v250; // rdi
  char *v251; // rbx
  const void *v252; // rdx
  signed __int64 v253; // rdi
  char *v254; // rcx
  __int64 v256; // rdx
  __int64 v257; // rdi
  __int64 v258; // rdx
  int v259; // r9d
  int v262; // esi
  __int64 v263; // rdi
  bool v265; // cc
  bool v286; // cf
  char v287; // zf
  unsigned __int64 v305; // rdx
  unsigned __int64 *v308; // rax
  __int64 v311; // rcx
  int v312; // edx
  unsigned __int64 *v313; // rax
  int v316; // ecx
  __int64 v317; // rcx
  __int64 v318; // rdx
  __int64 v319; // rdi
  __int64 v320; // rdx
  __int64 v321; // rdx
  __int64 v322; // rdi
  __int64 v323; // r13
  int v324; // r9d
  __int64 v325; // rdi
  char *v326; // r8
  int v327; // r12d
  __int64 v329; // r15
  __int64 v330; // rax
  __int64 v331; // rcx
  __int64 v340; // rdx
  __int64 v341; // rsi
  __int64 v343; // rbx
  _QWORD *v344; // rax
  __int64 v345; // rdx
  bool v347; // cc
  __int64 v348; // rbx
  __int64 v350; // rdi
  __int64 v355; // rax
  __int64 v356; // rax
  __int64 v357; // rdx
  __int64 v358; // rdx
  unsigned __int64 v371; // rdx
  __int128 *v372; // r12
  unsigned __int64 v373; // rdx
  __int64 v374; // rdi
  _DWORD *v375; // r15
  const void *v376; // rdx
  signed __int64 v377; // rdi
  char *v378; // rbx
  int v379; // r14d
  bool v381; // cc
  unsigned __int64 v387; // rdx
  __int64 v391; // r14
  unsigned __int64 v392; // rbx
  unsigned __int64 v393; // rsi
  __int64 v397; // rdi
  __int64 v400; // r15
  __time64_t v404; // rsi
  __time64_t v408; // rbx
  __int64 jj; // rdx
  __int64 v414; // rcx
  unsigned __int64 v415; // rax
  __int64 v417; // rsi
  unsigned __int64 v418; // rdx
  __int64 v419; // rdi
  signed __int64 v421; // rdi
  char *v422; // rcx
  unsigned int v423; // r13d
  __int64 v424; // rdx
  int *v425; // rsi
  int *v426; // rax
  int v427; // eax
  void *v428; // rax
  __int64 *v429; // rdx
  int v430; // r9d
  int v431; // r8d
  __m256 *v436; // r12
  __int64 v437; // r14
  unsigned __int64 v438; // rbx
  int *v439; // rsi
  unsigned __int64 v440; // rdi
  _QWORD *v441; // rax
  __int64 v444; // rdx
  __int64 v445; // rcx
  unsigned __int64 v446; // rax
  int InitFlagb; // [rsp+20h] [rbp-E0h]
  int InitFlag; // [rsp+20h] [rbp-E0h]
  int InitFlagc; // [rsp+20h] [rbp-E0h]
  int InitFlagd; // [rsp+20h] [rbp-E0h]
  int InitFlaga; // [rsp+20h] [rbp-E0h]
  int ThrdAddr; // [rsp+28h] [rbp-D8h]
  __int128 v464; // [rsp+50h] [rbp-B0h] BYREF
  __m256 *v465; // [rsp+60h] [rbp-A0h]
  __int64 v466; // [rsp+68h] [rbp-98h] BYREF
  int *v467; // [rsp+70h] [rbp-90h]
  _DWORD *v468; // [rsp+78h] [rbp-88h]
  __int128 *v469; // [rsp+80h] [rbp-80h]
  _QWORD v470[3]; // [rsp+88h] [rbp-78h] BYREF
  int v471; // [rsp+A0h] [rbp-60h] BYREF
  _Thrd_t v472; // [rsp+B0h] [rbp-50h] BYREF
  __int128 v473; // [rsp+C0h] [rbp-40h] BYREF
  char *v474; // [rsp+D0h] [rbp-30h]
  __int128 v475; // [rsp+D8h] [rbp-28h] BYREF
  __int64 v476; // [rsp+E8h] [rbp-18h]
  __int64 v477; // [rsp+F0h] [rbp-10h]
  __int128 v478; // [rsp+F8h] [rbp-8h] BYREF
  __int64 v479; // [rsp+108h] [rbp+8h]
  __int64 v480; // [rsp+110h] [rbp+10h]
  unsigned int v481[4]; // [rsp+120h] [rbp+20h] BYREF
  __int128 v482; // [rsp+130h] [rbp+30h] BYREF
  char *v483; // [rsp+140h] [rbp+40h]
  __int128 v484; // [rsp+148h] [rbp+48h] BYREF
  __int64 v485; // [rsp+158h] [rbp+58h]
  int v486; // [rsp+160h] [rbp+60h]
  int v487; // [rsp+164h] [rbp+64h] BYREF
  int v488; // [rsp+168h] [rbp+68h]
  int v489; // [rsp+16Ch] [rbp+6Ch] BYREF
  _QWORD v490[3]; // [rsp+170h] [rbp+70h] BYREF
  int v491; // [rsp+188h] [rbp+88h]
  int v492; // [rsp+18Ch] [rbp+8Ch] BYREF
  int v493; // [rsp+190h] [rbp+90h]
  int v494; // [rsp+194h] [rbp+94h] BYREF
  __int128 v495; // [rsp+198h] [rbp+98h] BYREF
  char *v496; // [rsp+1A8h] [rbp+A8h]
  _QWORD v497[3]; // [rsp+1B0h] [rbp+B0h] BYREF
  __int64 *v498; // [rsp+1C8h] [rbp+C8h]
  int *v499; // [rsp+1D0h] [rbp+D0h]
  char v500[24]; // [rsp+1D8h] [rbp+D8h] BYREF
  char v501[16]; // [rsp+1F0h] [rbp+F0h] BYREF
  char v502[96]; // [rsp+200h] [rbp+100h] BYREF
  char v503[96]; // [rsp+260h] [rbp+160h] BYREF
  char v504[144]; // [rsp+2C0h] [rbp+1C0h] BYREF
  char v505[16]; // [rsp+350h] [rbp+250h] BYREF
  char v506[96]; // [rsp+360h] [rbp+260h] BYREF
  char v507[96]; // [rsp+3C0h] [rbp+2C0h] BYREF
  char v508[144]; // [rsp+420h] [rbp+320h] BYREF
  int v509; // [rsp+4B0h] [rbp+3B0h] BYREF
  int v510; // [rsp+4B4h] [rbp+3B4h] BYREF
  int v511; // [rsp+4B8h] [rbp+3B8h] BYREF
  int v512; // [rsp+4BCh] [rbp+3BCh] BYREF
  _QWORD *v513; // [rsp+4C0h] [rbp+3C0h] BYREF
  _BYTE v514[4]; // [rsp+4C8h] [rbp+3C8h] BYREF
  int v515; // [rsp+4CCh] [rbp+3CCh] BYREF
  int v516; // [rsp+4D0h] [rbp+3D0h] BYREF
  int v517; // [rsp+4D8h] [rbp+3D8h] BYREF
  int v518; // [rsp+4E0h] [rbp+3E0h] BYREF
  int v519; // [rsp+4E4h] [rbp+3E4h] BYREF
  int v520; // [rsp+4E8h] [rbp+3E8h] BYREF
  __int128 v521; // [rsp+4F0h] [rbp+3F0h] BYREF
  char *v522; // [rsp+500h] [rbp+400h]
  int v523; // [rsp+508h] [rbp+408h] BYREF
  _QWORD v524[2]; // [rsp+510h] [rbp+410h] BYREF
  __int128 v525; // [rsp+520h] [rbp+420h] BYREF
  char *v526; // [rsp+530h] [rbp+430h]
  _QWORD v527[3]; // [rsp+538h] [rbp+438h] BYREF
  __time64_t Time2; // [rsp+550h] [rbp+450h] BYREF
  int v529; // [rsp+558h] [rbp+458h] BYREF
  __int128 v530; // [rsp+560h] [rbp+460h] BYREF
  _QWORD *v531; // [rsp+570h] [rbp+470h]
  unsigned __int64 v532[3]; // [rsp+578h] [rbp+478h] BYREF
  unsigned __int64 v533; // [rsp+590h] [rbp+490h] BYREF
  int *v534; // [rsp+598h] [rbp+498h]
  __int64 v535; // [rsp+5A0h] [rbp+4A0h]
  unsigned __int64 v536; // [rsp+5A8h] [rbp+4A8h] BYREF
  __int64 v537; // [rsp+5B0h] [rbp+4B0h]
  __m256 *v538; // [rsp+5B8h] [rbp+4B8h]
  __int128 v539; // [rsp+5C0h] [rbp+4C0h] BYREF
  __int64 v540; // [rsp+5D0h] [rbp+4D0h]
  unsigned __int64 v541; // [rsp+5D8h] [rbp+4D8h] BYREF
  unsigned __int64 v542; // [rsp+5E0h] [rbp+4E0h]
  __int128 v543; // [rsp+5E8h] [rbp+4E8h] BYREF
  char *v544; // [rsp+5F8h] [rbp+4F8h]
  __int128 v545; // [rsp+600h] [rbp+500h] BYREF
  _DWORD *v546; // [rsp+610h] [rbp+510h]
  __int128 v547; // [rsp+618h] [rbp+518h] BYREF
  __int64 v548; // [rsp+628h] [rbp+528h]
  _QWORD v549[3]; // [rsp+630h] [rbp+530h] BYREF
  __int128 v550; // [rsp+648h] [rbp+548h] BYREF
  _QWORD *v551; // [rsp+658h] [rbp+558h]
  __int128 v552; // [rsp+660h] [rbp+560h] BYREF
  _QWORD *v553; // [rsp+670h] [rbp+570h]
  __int128 v554; // [rsp+678h] [rbp+578h] BYREF
  _QWORD *v555; // [rsp+688h] [rbp+588h]
  _QWORD v556[3]; // [rsp+690h] [rbp+590h] BYREF
  __int128 v557; // [rsp+6A8h] [rbp+5A8h] BYREF
  _QWORD *v558; // [rsp+6B8h] [rbp+5B8h]
  int v559; // [rsp+6C0h] [rbp+5C0h] BYREF
  int v560; // [rsp+6C8h] [rbp+5C8h] BYREF
  __int64 v561[3]; // [rsp+6D0h] [rbp+5D0h] BYREF
  unsigned __int64 v562[3]; // [rsp+6E8h] [rbp+5E8h] BYREF
  __m256 v563; // [rsp+700h] [rbp+600h] BYREF
  __m256 v564; // [rsp+720h] [rbp+620h]
  __m256 v565; // [rsp+740h] [rbp+640h]
  _BYTE v566[64]; // [rsp+760h] [rbp+660h] BYREF
  __int128 v567; // [rsp+7A0h] [rbp+6A0h] BYREF
  __int64 v568; // [rsp+7B0h] [rbp+6B0h]
  __m256 v569; // [rsp+7C0h] [rbp+6C0h] BYREF
  __m256 v570; // [rsp+7E0h] [rbp+6E0h]
  __m256 v571; // [rsp+800h] [rbp+700h]
  int *v572; // [rsp+820h] [rbp+720h]
  int *v573; // [rsp+828h] [rbp+728h]
  __int128 *v574; // [rsp+830h] [rbp+730h]
  int *v575; // [rsp+838h] [rbp+738h]
  int *v576; // [rsp+840h] [rbp+740h]
  int *v577; // [rsp+848h] [rbp+748h]
  int *v578; // [rsp+850h] [rbp+750h]
  __int128 *v579; // [rsp+858h] [rbp+758h]
  unsigned __int64 *v580; // [rsp+860h] [rbp+760h]
  int *v581; // [rsp+868h] [rbp+768h]
  _QWORD *v582; // [rsp+870h] [rbp+770h]
  _BYTE v583[96]; // [rsp+880h] [rbp+780h] BYREF
  __m256 v584; // [rsp+8E0h] [rbp+7E0h] BYREF
  __m256 v586; // [rsp+920h] [rbp+820h]
  __m256 v587; // [rsp+940h] [rbp+840h] BYREF
  char v590; // [rsp+A58h] [rbp+958h] BYREF

  v25 = a12;
  v26 = a5;
  v523 = a5;
  v498 = a14;
  v499 = a15;
  v27 = 0;
  *a15 = 0;
  v486 = a9;
  v471 = a9;
  a9 = 0;
  if ( !a1 )
    goto LABEL_44;

  // Phase 0: 创建 cv::Mat 包装图像
  mwwz::region::__autoclassinit2((mwwz::region *)v583, 0x60u);
  *(double *)&_XMM0 = cv::Mat::Mat((cv::Mat *)v583, a3, a2, 0, a1, 0);
  v28 = 0;
  v29 = a5;
  Time2 = a5;

  // 验证所有模型句柄有效
  if ( a5 > 0 )
  {
    while ( sub_1800B9BE0(*(_DWORD *)(a4 + 4 * v28)) )
    {
      if ( ++v28 >= a5 )
        goto LABEL_5;
    }
    goto LABEL_43;
  }

LABEL_5:
  // Phase 1: 初始化逐模型数据结构
  v515 = 0;
  v518 = 255;
  mwwz::region::__autoclassinit2((mwwz::region *)v490, 0x18u);
  sub_180002830(v490, a5);          // alloc v490[a5] — 模型 maxLevel 数组
  mwwz::region::__autoclassinit2((mwwz::region *)v549, 0x18u);
  sub_180002830(v549, a5);          // alloc v549[a5] — 模型 offset+64 数组
  mwwz::region::__autoclassinit2((mwwz::region *)v470, 0x18u);
  v519 = 0;
  sub_1800217A0(v470, 2 * a5, &v519); // alloc v470[2*a5] — numLevels + startLevel
  sub_180014340((__int64)v527);
  sub_18007CF60(v527, a5);          // 模型对象数组
  sub_18007B820(v566);              // 模型 ID → index 映射表
  sub_18007CEE0(v566);
  sub_18007BB90(v524);              // 模型排序容器
  sub_18007CCE0(v524);
  v30 = 0;

  // xmm7 = arg_30 (angleStart), xmm8 = arg_28 (angleExtent)
  __asm
  {
    vmovss  xmm7, [rbp+990h+arg_30]
    vmovss  xmm8, [rbp+990h+arg_28]
  }

  if ( a5 <= 0 )
    goto LABEL_20;

  // 逐模型提取属性
  v513 = (_QWORD *)(v490[0] - 4LL * a5);
  do
  {
    // 注册模型 ID → index 映射
    *(_DWORD *)sub_18007CEC0(v566, a4 + 4LL * v27) = v27;

    // 获取模型对象
    v33 = v527[0];
    *(_QWORD *)(v33 + 8 * v30) = sub_18007CD50(v34, (int *)(a4 + 4LL * v27));

    // 设置模型搜索参数 (angleStart, angleExtent)
    _RCX = *(_QWORD *)(v527[0] + 8 * v30);
    __asm { vmovss  dword ptr [rcx+4], xmm8 }   // model.angleExtent = arg_28
    _RCX = *(_QWORD *)(v527[0] + 8 * v30);
    __asm { vmovss  dword ptr [rcx+8], xmm7 }   // model.angleStart = arg_30

    v29 = Time2;
    v37 = 4 * Time2 + 4 * v30;

    // 提取模型属性到各数组
    *(_DWORD *)((char *)v513 + v37) = *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 72LL); // maxLevel (offset+72)
    *(_DWORD *)(v549[0] + 4 * v30) = *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 64LL); // offset+64
    *(_DWORD *)(v470[0] + 4 * v30) = **(_DWORD **)(v527[0] + 8 * v30);                   // numLevels (offset+0)

    // 跟踪全局最小/最大值
    if ( *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 88LL) < v518 )
      v518 = *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 88LL); // min across models (offset+88)
    if ( *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 72LL) > v515 )
      v515 = *(_DWORD *)(*(_QWORD *)(v527[0] + 8 * v30) + 72LL); // max across models (offset+72)

    // a12 提供逐模型 (numLevels, startLevel) 覆盖
    v38 = *(_DWORD *)(v25 + 8 * v30);   // a12[model*2] = numLevels override
    v39 = v470[0];
    if ( v38 > 0 && v38 <= *(_DWORD *)(v470[0] + 4 * v30) )
    {
      *(_DWORD *)(v470[0] + 4 * v30) = v38;
      v39 = v470[0];
    }

    v40 = *(_DWORD *)(v39 + 4 * v30);
    if ( v40 <= 0 )
    {
      // 模型 numLevels 无效 → 清理并返回 -1
      goto LABEL_43;
    }

    v41 = *(_DWORD *)(v25 + 8 * v30 + 4); // a12[model*2+1] = startLevel override
    if ( v41 > 0 && v41 <= v40 )
    {
      *(_DWORD *)(v39 + v37) = v41 - 1;   // startLevel (0-based)
      v39 = v470[0];
    }

    sub_18007CB20(v524, &v472, v39 + 4LL * v27++);
    ++v30;
  }
  while ( v30 < v29 );
  v26 = v29;

LABEL_20:
  // v518 = 全局最小值, clamp ≥ 1
  v42 = 1;
  if ( v518 > 1 )
    v42 = v518;
  v518 = v42;

  // v510 = 共享金字塔总层数
  v510 = *(_DWORD *)(*(_QWORD *)v524[0] + 28LL);

  // 分配逐模型 startLevel / stopLevel 数组
  mwwz::region::__autoclassinit2((mwwz::region *)v556, 0x18u);
  sub_180002830(v556, v29);   // v556[modelCount] — stopLevel per model
  mwwz::region::__autoclassinit2((mwwz::region *)v497, 0x18u);
  sub_180002830(v497, v29);   // v497[modelCount] — numLevels per model

  // 分配逐模型逐层的激活标志 v561[model][level]
  sub_180014340((__int64)v561);
  v519 = 0;
  v43 = sub_1800217A0(&v464, v510, &v519);
  sub_180021650(v561, v29, v43);
  sub_1800027C0((__int64)&v464);

  // 初始化逐模型 level 激活标志
  v44 = 0;
  if ( v29 > 0 )
  {
    v45 = 4 * v29;
    v46 = v497[0] - 4 * v29;
    v47 = -v29;
    v48 = v29;
    v49 = 0;
    v50 = v470[0];
    do
    {
      *(_DWORD *)(v45 + v46) = *(_DWORD *)(v50 + v49); // v497[m] = numLevels[m]
      v51 = *(_DWORD *)(v470[0] + v45);
      v52 = v45 + 4 * v47;
      if ( v51 < 1 )
        v51 = 1;
      *(_DWORD *)(v556[0] + v52) = v51;               // v556[m] = max(numLevels[m+modelCount], 1) = startLevel

      v50 = v470[0];
      v53 = *(int *)(v470[0] + v52);                   // startLevel for this model
      v54 = *(_DWORD *)(v470[0] + v45);                // numLevels for this model
      if ( v54 < 1 )
        v54 = 1;

      // 对每个激活的层设置标志=1
      if ( (int)v53 >= v54 )
      {
        v55 = v44;
        v56 = 4 * v53;
        do
        {
          *(_DWORD *)(v56 + *(_QWORD *)sub_180021630(v561, v55) - 4) = 1;
          v58 = v57 - 1;
          v56 -= 4;
          v50 = v470[0];
          v59 = *(_DWORD *)(v470[0] + v45);
          if ( v59 < 1 )
            v59 = 1;
        }
        while ( v58 >= v59 );
      }
      ++v44;
      v45 += 4;
      --v48;
      v49 += 4;
    }
    while ( v48 );
    v26 = v523;
  }

  // Phase 2: 计时器 + 共享金字塔构建
  sub_1800B5890(&v541);
  v542 = v541;
  sub_180014340((__int64)&v539);    // 金字塔数据数组

  // 构建共享图像金字塔 (v510 层)
  for ( i = 0; i < v510; ++i )
  {
    if ( i > 0 )
    {
      sub_18000B3F0((__int64)&v464, (__int64)v583);
      sub_18000B3D0((__int64)&v478, (__int64)v583);
      v513 = 0;
      cv::pyrDown(&v478, &v464, &v513, 4);  // 2× 下采样
    }
    *(double *)_XMM0.m128_u64 = sub_18007E2F0((__int64)&v539, (const struct cv::Mat *)v583);
  }

  // 性能检查: 如果金字塔构建超时则提前返回
  // ... (timing check with qword_1800D6BD8)

  // Phase 3: angleStep 表填充
  // v532[level] = arg_38 * dword_1800D4D50[level]
  // (展开循环, 4 路并行)
  v66 = v510 - 1;
  _RCX = v510 - 1;
  if ( _RCX >= 4 )
  {
    // SIMD 4-way unrolled
    do
    {
      __asm { vmulss  xmm0, xmm2, ds:rva dword_1800D4D50[r8+rdx*4] }
      // v532[level] = angleStep * multiplier[level]
      // ... (4 elements per iteration)
    }
    while ( _RCX > 3 );
  }
  // remainder loop for last few levels

  // Phase 4: 逐模型最粗层搜索
  // 分配候选数组: v530[level] = vector of candidates
  // 分配 v554, v552 等辅助结构
  // ...

  v109 = 0;
  v512 = 0;
  if ( v26 > 0 )  // v26 = modelCount
  {
    do
    {
      // 找当前模型的最粗有效层
      v115 = (unsigned int)(v510 - 1);
      v509 = v510 - 1;
      while ( *(int *)(*(_QWORD *)(v561[0] + 8 * v116) + 4 * v115) <= 0 )
      {
        v115 = (unsigned int)(v117 - 1);
        if ( (int)v115 < 0 )
          goto LABEL_123;
      }

      // 提取当前层的图像尺寸
      v120 = *(_DWORD *)(v119 + v539 + 8);   // width at level
      v121 = *(_DWORD *)(v119 + v539 + 12);  // height at level

      // 获取模型对象
      v122 = *(_QWORD **)(v527[0] + 8LL * v109);
      v513 = v122;

      // 如果 v554 (角度搜索网格) 未初始化, 生成初始搜索参数
      if ( !((__int64)(*((_QWORD *)v123 + 1) - *(_QWORD *)v123) >> 2) )
      {
        // 从模型提取 6 个浮点参数 (offset 18,24,21,27,30,33 × 4)
        // 调用 sub_18007DFF0 初始化搜索范围
      }

      // 生成搜索 ROI (sub_1800073E0 + sub_180007930)
      if ( !((v137[1] - *v137) / 96LL) )
      {
        sub_1800073E0(v539 + 96LL * v509, ...);  // 搜索区域生成
        sub_180007930((__int64)&v563, (__int64)v137);
      }

      // angleStep 计算
      // if (level+1 == model.numLevels) factor = dword_1800D6ADC (0.8)
      // else factor = dword_1800D6B38
      // angleStep = acos(1.0 - factor²/(2*R*R²)) * 180/π
      // angleStep = min(qword_1800D6BA0, angleStep)

      // sub_1800B7FB0: 生成搜索候选位置
      sub_1800B7FB0(_RBX, (const void **)&v525, _EDI, *(double *)&_XMM3, InitFlagb);

      // 分配候选结果容器 v521
      // ...

      // PPL 并行搜索
      v514[0] = 0;  // error flag
      // 打包搜索上下文到 v569/v570/v571/v572-v578
      *(_QWORD *)v569.m256_f32 = v514;      // error flag
      *(_QWORD *)&v569.m256_f32[2] = &v525; // 候选位置
      *(_QWORD *)&v569.m256_f32[4] = &v521; // 输出结果
      // ... (更多上下文参数)

      LOBYTE(v159) = 4;
      Concurrency::_Trace_ppl_function(*(_QWORD *)&Concurrency::PPLParallelForEventGuid.Data1, v159, 1);
      if ( (int)v164 > 0 )
      {
        if ( (unsigned int)v164 > 1 )
          sub_18008D9A0(&v517, (unsigned int)v164, &v529, &v569); // PPL parallel_for
        else
          sub_180067920(&v569, 0);  // 单任务直接执行
      }

      // 标记每个结果的 modelID
      v169 = 0;
      do
      {
        v170 = *(char **)(v169 + v168);
        if ( (*(_QWORD *)(v169 + v168 + 8) - (_QWORD)v170) / 40LL )
        {
          v171 = 0;
          v172 = 0;
          do
          {
            *(_DWORD *)&v170[v172 + 8] = *(_DWORD *)(a4 + 4LL * v512); // result.modelID = a4[modelIndex]
            ++v171;
            v172 += 40;  // 每个结果 40 字节
          }
          while ( ... );

          // 追加到全局候选 v484
          sub_18007FD70(&v484, ...);
        }
        ++v166;
        v169 += 24;
      }
      while ( v166 < ... );

      // 最终层调度 (与单模型相同)
      if ( v509 == *(_DWORD *)(v556[0] + 4LL * v512) - 1 )
      {
        // stopLevel 已到 → 调用最终细化
        sub_18004C8C0(a9, v520, v519, v173, InitFlagc, &v484, (__int64)v513, v509);
      }
      else
      {
        // 中间层 → 调用中间细化
        sub_18004A5A0(...);
      }

      // 保存到 v530[level]
      if ( (*((_QWORD *)&v484 + 1) - (_QWORD)v484) / 40LL )
        sub_18007FD70(v530 + 24LL * v509, ...);

      ++v512;
    }
    while ( v109 < v26 );
  }

  // Phase 5: 金字塔细化 (从粗到细, 所有模型的候选混合)
  v218 = v511;       // stopLevel (全局最小)
  v219 = v510 - 2;   // 从第二粗层开始
  if ( v510 - 2 >= v218 )
  {
    while ( 1 )
    {
      // 计算当前层的搜索半径
      // v523 = max(scoreGrid[model]) across all models + optional padding

      // 如果有缩放路径 (v520 > 0):
      if ( *(int *)(v543 + 4LL * v219) > 1 )
      {
        v520 = 1;
        // 逐模型生成缩放搜索网格
        do
        {
          v238 = v219 + v510 * v236;
          v239 = (_QWORD *)(v550 + 24 * v238);
          if ( *v239 == v239[1] )
          {
            sub_1800B7E20(...);   // 生成缩放候选
            sub_18007C880(...);   // 变换
            sub_1800B54A0(...);   // 保存
          }
          ++v236;
        }
        while ( v237 < n );

        // 有缩放: 准备层数据
        sub_180036DC0(v539 + 96LL * v219, ...);
      }
      else
      {
        // 无缩放: 准备层数据
        sub_180037E10(v539 + 96LL * v219, ..., _EDI);
      }

      // 收集上一层传下来的候选
      // (从 v530[level+1] 拷贝到 v521)

      // PPL 并行细化 (跨模型候选统一并行)
      v514[0] = 0;
      // 打包上下文...
      if ( (int)v257 > 0 )
      {
        if ( (unsigned int)v257 > 1 )
          sub_18008DD40(&v517, (unsigned int)v257, &v516, &v569); // PPL parallel_for
        else
          sub_180067BA0(&v569, 0);
      }

      // 角度范围调整 + 候选过滤
      // score < v532[level] 的移除 (memmove 紧缩)
      do
      {
        _RAX = v532[0];
        __asm { vmovss  xmm0, dword ptr [rax+rcx*4] }  // threshold = v532[level]
        __asm { vcomiss xmm0, dword ptr [r8+rbx+18h] }  // compare with candidate.score
        if ( v265 )
        {
          // 保留: 调整角度到有效范围
          // ... (角度 wrap-around 逻辑)
          ++v262;
          _RBX += 40;
        }
        else
        {
          // 移除: memmove 紧缩数组
          memmove(...);
          *((_QWORD *)&v521 + 1) -= 40LL;
        }
      }
      while ( v262 < v305 );

      // 最终层调度
      if ( v259 > v511 )
      {
        // 中间层 (stopLevel 以上)
        sub_18004D9C0(...);
      }
      else
      {
        // 最终层
        sub_18004D280(a9, v312, ..., v259);
      }

      // SubPixel mode 1 (parabolic)
      if ( v316 == v511 && a11 == 1 )
      {
        if ( v520 )  // 有缩放
          sub_180068170(&v563, 0);   // or sub_18008E480 (parallel)
        else          // 无缩放
          sub_180068030(&v563, 0);   // or sub_18008E0E0 (parallel)
      }

      // SubPixel mode 2 (least-squares, 两次迭代)
      if ( v316 == v511 && a11 == 2 )
      {
        v324 = 0;
        v519 = 0;
        do
        {
          // 逐候选: 查找所属模型 → 生成搜索网格 → cv::solve 最小二乘
          // 更新 row, col 亚像素偏移和角度修正
          // ... (详见原始代码)
          v519 = ++v324;
        }
        while ( v324 < 2 );  // 两次迭代
      }

      // 保存到 v530[level]
      if ( v371 )
        sub_18007FD70(v530 + 24LL * v316, ...);

      v509 = v316 - 1;
      v219 = v509;
      if ( v509 < v511 )
        break;
    }
  }

  // Phase 6: stopLevel 输出缩放
  if ( v218 > 0 )
  {
    // 拷贝 v530[stopLevel] 到 v495
    // 逐候选:
    //   score < arg_38 → 移除
    //   坐标 *= 2^stopLevel
    //   亚像素偏移 *= 2^stopLevel
    v379 = 0;
    while ( v379 < v387 )
    {
      if ( arg_38 <= candidate.score )
      {
        *(_RDI - 1) *= 1 << v218;     // row *= 2^stopLevel
        *_RDI *= 1 << v511;           // col *= 2^stopLevel
        // subpixel offsets *= 2^level
        ++v379;
      }
      else
      {
        // 移除低分候选
        memmove(...);
      }
    }
  }

  // Phase 7: 结果输出
  v423 = 0;
  v424 = (v422 - _RBX) / 40;    // result count
  v425 = v499;                    // a15 = output count
  *v499 = v424;

  if ( (int)v424 > 0 )
  {
    // numMatches 截断
    if ( v486 > 0 )
    {
      if ( v486 >= (int)v424 )
        v427 = v424;
      else
        v427 = v486;
      *v425 = v427;
    }

    // 分配输出数组 (40 bytes per result)
    v428 = (void *)sub_1800C6ED4(saturated_mul((int)v424, 0x28u));
    *v498 = (__int64)v428;  // a14 = output array

    // 拷贝结果 (32 bytes ymm + 8 bytes tail)
    v430 = 0;
    do
    {
      __asm
      {
        vmovups ymm0, ymmword ptr [rbx+rdi]       // 32 bytes
        vmovups ymmword ptr [rax+rdi], ymm0
        vmovsd  xmm1, qword ptr [rbx+rdi+20h]     // 8 bytes (offset 0x20)
        vmovsd  qword ptr [rax+rdi+20h], xmm1
      }
      *(_DWORD *)(*v429 + _RDI + 20) = *(_DWORD *)(*v429 + _RDI + 16); // copy score
      ++v430;
      _RDI += 40;
    }
    while ( v430 < *v425 );

    // 排序 (sub_1800B9C20)
    v423 = sub_1800B9C20(0, *v429, v431);
  }

  // Phase 8: 后台清理线程
  // beginthreadex(sub_1800831E0) 释放金字塔等资源
  // ...

  return result;
}
```

</details>

---

## 0. 函数签名与参数映射

### 0.1 find_shape_models 签名

```c
__int64 __fastcall find_shape_models(
    void *a1,          // rcx: 图像数据指针 (cv::Mat data)
    int   a2,          // edx: 图像列数 (cols)  — cv::Mat::Mat(v583, a3, a2, 0, a1, 0)
    int   a3,          // r8:  图像行数 (rows)  — 同上
    __int64 a4,        // r9:  模型句柄数组 (int32[a5]) — *(_DWORD *)(a4 + 4 * v28) 遍历
    int   a5,          // +20h: 模型数量 — if (a5 > 0) 模型循环
    float a6,          // +28h (xmm8): AngleExtent — vmovss dword ptr [rcx+4], xmm8 → model+4
    float a7,          // +30h (xmm7): AngleStart  — vmovss dword ptr [rcx+8], xmm7 → model+8
    float a8,          // +38h: MinScore — v532[l] = a8 * dword_1800D4D50[l] 逐层阈值
    int   a9,          // +40h: NumMatches — v486 = a9; v471 = a9; 结果数限制
    float a10,         // +48h: MaxOverlap — [rbp+990h+arg_48] 传给 sub_18004C8C0 (GreedyNMS)
    int   a11,         // +50h: SubPixel — a11 % 10 = mode, clamp 0-3
    __int64 a12,       // +58h: NumLevels 数组 — v25 = a12, 逐模型 (numLevels, startLevel) 对
    int   a13,         // +60h: Greediness — 填充 v545 逐层 greediness 数组
    __int64 *a14,      // +68h: Output ptr — *v498 = v428 输出结果指针
    int  *a15          // +70h: Output count — *a15 = 0 初始化, *v425 = v424 设置结果数
);
```

### 0.2 与 find_shape_model (单数) 的参数对比

| 参数 | find_shape_model (14参) | find_shape_models (15参) | 差异 |
|------|------------------------|--------------------------|------|
| a1-a3 | image data, cols, rows | image data, cols, rows | 同 |
| a4 | model ID (单个 int32) | **model ID 数组** (int32[a5]) | **数组** |
| a5 | [float] angleStart | **模型数量** (int) | **新增** |
| a6 | [float] angleExtent | **[float] AngleExtent** (xmm8) | 位移 |
| a7 | [float] angleStep | **[float] AngleStart** (xmm7) | 位移 |
| a8 | [float] minScore | **[float] MinScore** | 同语义 |
| a9 | maxMatches | **NumMatches** (int) | 同语义 |
| a10 | subPixel + searchRadius×10 | **[float] MaxOverlap** | **不同** |
| a11 | [numLevels, startLevel] | **SubPixel** (int, %10=mode, clamp 0-3) | **不同编码** |
| a12 | greediness | **逐模型 (numLevels, startLevel) 对数组** | **关键差异** |
| a13 | output results ptr | **Greediness** (int) | 位移 |
| a14 | output count | **output results ptr** | 位移 |
| a15 | -- | **output count** | **新增** |

**核心差异**:
1. a4 从单个 modelID 变为 modelID 数组，a5 为数组长度
2. a6/a7 顺序与单模型不同: a6=AngleExtent (xmm8), a7=AngleStart (xmm7)
3. a10 从 subPixel 编码变为 MaxOverlap (传给 GreedyNMS)
4. a12 提供逐模型的 `(numLevels, startLevel)` 对 -- 每个模型可以有不同的金字塔层数和终止层

### 0.3 Halcon 文档参数推测

根据 Halcon `find_shape_models` 文档参数顺序:

```
find_shape_models(Image, ModelIDs, AngleStart, AngleExtent,
                  MinScore, NumMatches, MaxOverlap, SubPixel,
                  NumLevels, Greediness,
                  Row, Column, Angle, Score, Model)
```

**已确认映射** (2026-03-11, 基于完整 IDA 反编译代码):

| Halcon 参数 | 反编译参数 | 位置 | 类型 | 证据 |
|------------|-----------|------|------|------|
| Image | a1, a2, a3 | rcx, edx, r8 | void*, int, int | `cv::Mat::Mat(v583, a3, a2, 0, a1, 0)` |
| ModelIDs | a4 | r9 | __int64 | `*(_DWORD *)(a4 + 4 * v28)` 遍历模型 |
| (model count) | a5 | +20h | int | `if (a5 > 0)` 模型数循环 |
| AngleExtent | a6 | +28h (xmm8) | float | `vmovss dword ptr [rcx+4], xmm8` -> model+4 |
| AngleStart | a7 | +30h (xmm7) | float | `vmovss dword ptr [rcx+8], xmm7` -> model+8 |
| MinScore | a8 | +38h | float | `v532[l] = a8 * dword_1800D4D50[l]` 逐层阈值 |
| NumMatches | a9 | +40h | int | `v486 = a9; v471 = a9;` 结果数限制 |
| MaxOverlap | a10 | +48h | float | `[rbp+990h+arg_48]` 传给 sub_18004C8C0 (GreedyNMS) |
| SubPixel | a11 | +50h | int | `a11 % 10`, clamp 0-3 |
| NumLevels | a12 | +58h | __int64 | `v25 = a12`, 逐模型 (numLevels, startLevel) 对 |
| Greediness | a13 | +60h | int | 填充 v545 逐层 greediness 数组 |
| (output ptr) | a14 | +68h | __int64* | `*v498 = v428` 输出结果指针 |
| (output count) | a15 | +70h | int* | `*a15 = 0` 初始化, `*v425 = v424` 设置结果数 |

> **注意**: a6=AngleExtent, a7=AngleStart 的顺序与 Halcon 文档 (AngleStart, AngleExtent) 相反。这是 x64 Windows calling convention 下浮点参数通过 xmm 寄存器传递的结果 -- xmm8 先于 xmm7 被存储。

---

## 1. Phase 0: 初始化与模型属性提取

### 1.1 输入验证

```c
// 基本检查
if (!a1) return ERROR;  // 图像数据非空
// 遍历 a4[0..a5-1] 验证每个模型句柄有效
for (int i = 0; i < a5; i++) {
    sub_1800B9BE0(a4[i]);  // 验证模型句柄, 类似 ShapeModel.IsValid()
}
```

### 1.2 图像包装

```c
// 创建 cv::Mat 包装图像 (与单模型版本相同)
cv::Mat imageMat(a3, a2, CV_8UC1, a1);
```

### 1.3 逐模型属性提取

```c
// v527: 模型对象数组
// v470: numLevels per model (从模型 offset 0 读取)
// v549: 模型 offset+64 的值
// v490: 模型 offset+72 的值 (最大 level)
// v518: 所有模型中最小的某值 (offset+88), 最终 clamp >= 1
// v515: 所有模型中最大的某值 (offset+72)

for (int i = 0; i < a5; i++) {
    model = sub_1800806A0(a4[i]);     // 根据 ID 查找模型对象
    v527[i] = model;
    v470[i] = *(int*)(model + 0);     // numLevels
    v549[i] = *(int*)(model + 64);    // 待确认
    v490[i] = *(int*)(model + 72);    // 最大 level

    // 全局极值
    v518 = min(v518, *(int*)(model + 88));  // 所有模型中最小值
    v515 = max(v515, *(int*)(model + 72));  // 所有模型中最大值
}
v518 = max(v518, 1);  // clamp >= 1

// a12 提供逐模型的 (numLevels, startLevel) 对
for (int i = 0; i < a5; i++) {
    int userNumLevels = ((int*)a12)[2*i];
    int userStartLevel = ((int*)a12)[2*i + 1];
    // 如果 userNumLevels == 0, 使用模型自带的 numLevels
}
```

### 1.4 全局参数计算

```c
int v510 = 最大 numLevels (所有模型中);    // 共享金字塔层数
int v511 = 最小 startLevel (所有模型中);   // 全局终止层
```

---

## 2. Phase 1: 共享图像金字塔构建

### 2.1 金字塔循环

```c
// v510 = 最大 numLevels (覆盖所有模型)
for (int i = 0; i < v510; i++) {
    if (i > 0) {
        cv::pyrDown(pyramid[i-1], pyramid[i]);
    }
    // sub_18007E2F0: 提取/缓存每层特征 (Sobel 梯度 + 响应图)
    sub_18007E2F0(pyramid[i], ...);
}
```

**关键设计**: 所有模型共享同一组金字塔。金字塔层数取所有模型的最大值，保证每个模型都有足够的层可用。这是 `find_shape_models` 相比逐个调用 `find_shape_model` 的主要性能优势。

### 2.2 与单模型的对比

| 特性 | find_shape_model | find_shape_models |
|------|-----------------|-------------------|
| 金字塔构建 | 每次调用构建 | 构建一次，共享 |
| 金字塔层数 | 单模型的 numLevels | max(所有模型的 numLevels) |
| 特征缓存 | 单模型使用 | 所有模型共享 |

---

## 3. Phase 2: angleStep 表填充

### 3.1 per-level angleStep

```c
// v532[level] = arg_38 * dword_1800D4D50[level]
// dword_1800D4D50 = {0.8f, 0.9f x 15} -- 已确认
for (int level = 0; level < v510; level++) {
    v532[level] = angleStep_user * dword_1800D4D50[level];
}
```

**与单模型完全相同**: 使用相同的常量表 `dword_1800D4D50` 进行逐层角度步长缩放。

### 3.2 angleStep 计算 (最粗层)

```c
// 最粗层的角度步长使用特殊公式
if (level + 1 == model.numLevels)
    factor = dword_1800D6ADC;   // 0.8 (已确认)
else
    factor = dword_1800D6B38;   // 2.0f (已确认)

double R = model_radius;        // 模型半径 (最远点到中心距离)
angleStep_rad = acos(1.0 - factor*factor / (2.0 * R * R)) * qword_1800D6BA8;
// qword_1800D6BA8 = 180/PI (rad->deg 转换)

angleStep_rad = min(qword_1800D6BA0, angleStep_rad);  // 上限 clamp
```

**与单模型完全相同**: 同一套角度步长计算逻辑。

---

## 4. Phase 3: 逐模型最粗层搜索

### 4.1 外层模型循环

```c
for (int v512 = 0; v512 < a5; v512++) {  // a5 = modelCount
    // 找到当前模型的最粗有效层
    int v509 = model_coarsest_level[v512];

    // 检查 v554 (候选数组) 是否已初始化
    if (v554[v512] == nullptr) {
        // 初始化搜索参数
        sub_1800073E0(...);  // 生成角度搜索网格
        sub_180007930(...);  // 某种变换操作
    }
```

### 4.2 候选生成

```c
    // 生成搜索候选位置 (与单模型共享同一函数)
    sub_1800B7FB0(angleStart, angleExtent, angleStep, ...);
```

### 4.3 PPL 并行搜索

```c
    // 根据任务量选择单任务/多任务
    if (candidate_count <= THRESHOLD) {
        sub_180067920(...);   // 单任务搜索
    } else {
        sub_18008D9A0(...);   // PPL 并行搜索 (Concurrency::parallel_for)
    }
```

### 4.4 结果标记 modelID

```c
    // **关键**: 每个候选结果标记所属模型 ID
    for (int j = 0; j < result_count; j++) {
        v170[v172 + 8] = a4[v512];  // a4[v512] = 当前模型的句柄 ID
    }
}
```

**与单模型的核心差异**: 结果中嵌入了 modelID 字段，使得后续处理可以区分匹配来自哪个模型。

### 4.5 粗搜索候选上限

与单模型相同，粗搜索候选数上限 500 (sort + truncate)。

---

## 5. Phase 4: 金字塔细化 (从粗到细)

### 5.1 细化循环结构

```c
// v509 从 v510-2 递减到 v511 (stopLevel)
for (int v509 = v510 - 2; v509 >= v511; v509--) {

    // 5.1 收集上一层传下来的候选
    // v530 = 来自上一层的候选列表 (混合所有模型的候选)

    // 5.2 缩放路径检查
    if (v520 > 0) {  // 有缩放维度
        sub_1800B7E20(...);   // 生成缩放搜索网格
        sub_18007C880(...);   // 变换操作
        sub_1800B54A0(...);   // 变换操作
    }

    // 5.3 准备本层数据 (梯度计算)
    if (v520 > 0) {
        sub_180036DC0(...);   // 有缩放时的层数据准备
    } else {
        sub_180037E10(...);   // 无缩放时的层数据准备
    }
```

### 5.2 PPL 并行细化

```c
    // **关键**: 跨模型的候选合并后统一并行细化
    if (candidate_count <= THRESHOLD) {
        sub_180067BA0(...);   // 单任务细化
    } else {
        sub_18008DD40(...);   // PPL 多任务并行细化
    }
```

**与单模型的差异**: 候选来自多个模型，但在同一个 parallel_for 中统一处理。每个候选自带 modelID，细化函数根据 modelID 选择对应模型的模板数据。

### 5.3 角度范围调整

```c
    // 角度 wrap-around 到 [angleStart, angleStart + angleExtent]
    for (each candidate) {
        while (angle < angleStart) angle += 2*PI;
        while (angle > angleStart + angleExtent) angle -= 2*PI;
    }
```

### 5.4 候选过滤

```c
    // score < v532[level] 的移除
    filter(candidates, [&](auto& c) { return c.score >= v532[v509]; });
```

### 5.5 最终层处理

```c
    if (v509 == v511) {  // 到达最终层
        if (v520 > 0) {  // 有缩放
            sub_18004D9C0(...);   // 缩放细化
        } else {
            sub_18004D280(...);   // 非缩放细化
            // 传入 arg_48 = greediness (来自 a10)
        }
    }
}
```

---

## 6. Phase 4 (续): SubPixel 处理

### 6.1 SubPixel 模式解码

```c
// a11 编码: %10 = subPixel mode, /10 = thread count
int subPixelMode = a11 % 10;
int threadCount = a11 / 10;
if (threadCount > 32) threadCount = 32;
```

### 6.2 SubPixel Mode 1: Parabolic (sub_180068030 / sub_180068170)

```c
if (subPixelMode == 1) {
    if (v520 > 0) {  // 有缩放
        sub_180068170(...);  // 缩放版 parabolic
    } else {
        sub_180068030(...);  // 非缩放版 parabolic
    }
}
```

与单模型 `sub_18005B7E0` 等效。

### 6.3 SubPixel Mode 2: Least-Squares (sub_18005B950 / sub_18005BE10)

```c
if (subPixelMode == 2) {
    // 两次迭代:
    for (int iter = 0; iter < 2; iter++) {
        sub_18005B950(...);   // 候选评估
        sub_18005BE10(...);   // LS 拟合 (cv::solve)
        // 更新 row, col 的亚像素偏移和角度修正
    }
}
```

- 使用 Bresenham 线段采样 +-5 步
- cv::solve 做最小二乘拟合
- 与单模型完全相同的子函数

---

## 7. Phase 5: stopLevel 输出缩放

```c
// 如果 v511 (stopLevel) > 0, 坐标需要缩放回原始分辨率
if (v511 > 0) {
    double scaleFactor = (double)(1 << v511);  // 2^stopLevel
    for (each result) {
        result.row *= scaleFactor;
        result.col *= scaleFactor;
        // 亚像素偏移同步缩放
        result.subRow *= scaleFactor;
        result.subCol *= scaleFactor;
    }

    // score < minScore (arg_38) 的结果过滤
    filter(results, [&](auto& r) { return r.score >= minScore; });
}
```

**与单模型相同**: stopLevel > 0 时的坐标缩放逻辑完全一致。

---

## 8. Phase 6: 结果输出

### 8.1 结果格式 (40 字节结构, 已确认 2026-03-11)

从完整 IDA 反编译代码的输出拷贝和 SubPixel 读写操作确认:

```c
// 输出拷贝: 40 字节
vmovups ymm0, ymmword ptr [rbx+rdi]       // bytes 0-31
vmovups ymmword ptr [rax+rdi], ymm0
vmovsd  xmm1, qword ptr [rbx+rdi+20h]     // bytes 32-39
vmovsd  qword ptr [rax+rdi+20h], xmm1
*(_DWORD *)(*v429 + _RDI + 20) = *(_DWORD *)(*v429 + _RDI + 16);  // 复制 offset 16 到 offset 20
_RDI += 40;
```

**完整 40 字节结构**:

| 偏移 | 大小 | 类型 | 字段 | IDA 证据 |
|------|------|------|------|----------|
| 0x00 | 4 | int32 | row (像素坐标) | `*_R14 = _EAX` (LS solve 写回), `*= 1 << level` (stopLevel 缩放) |
| 0x04 | 4 | int32 | col (像素坐标) | `_R14[1] = _RCX` (LS solve 写回), `*= 1 << level` |
| 0x08 | 4 | int32 | modelID | `*(_DWORD *)&v170[v172+8] = *(_DWORD *)(a4 + 4LL * v512)` |
| 0x0C | 4 | float | angle (弧度) | `vmovss xmm3, [r14+0Ch]` (SubPixel 读取), LS 后更新 |
| 0x10 | 4 | float | scale | `vmovss xmm0, [r14+10h]` (SubPixel 读取) |
| 0x14 | 4 | float | scale (输出时复制自 0x10) | `dest[20]=dest[16]` -- 非缩放时两者相同 |
| 0x18 | 4 | float | score | `vcomiss xmm6, [rdi+14h]` (from _RDI=result+4, 即 offset 0x18) |
| 0x1C | 4 | float | (保留/未使用) | 未见读写 |
| 0x20 | 4 | float | subpixel row delta | `vaddss xmm0, [r14], [r14+20h]` (SubPixel 合成), LS 后写回 |
| 0x24 | 4 | float | subpixel col delta | `vaddss xmm2, [r14+4], [r14+24h]` (SubPixel 合成), LS 后写回 |

**SubPixel mode 2 读写操作详情**:

```c
// 读取 (SubPixel mode 2 LS 前):
vcvtsi2ss xmm0, xmm0, dword ptr [r14]      // offset 0: row (int->float)
vaddss  xmm0, xmm0, dword ptr [r14+20h]     // offset 0x20: + subpixel row delta
vcvtsi2ss xmm1, xmm1, dword ptr [r14+4]     // offset 4: col (int->float)
vaddss  xmm2, xmm1, dword ptr [r14+24h]     // offset 0x24: + subpixel col delta
vmovss  xmm3, dword ptr [r14+0Ch]           // offset 0x0C: angle
vmovss  xmm0, dword ptr [r14+10h]           // offset 0x10: scale

// 写回 (LS solve 后):
*_R14 = _EAX;                               // offset 0: new row (int)
_R14[1] = _RCX;                             // offset 4: new col (int)
vmovss  dword ptr [r14+20h], xmm1           // offset 0x20: new subpixel row
vmovss  dword ptr [r14+24h], xmm0           // offset 0x24: new subpixel col
vmovss  dword ptr [r14+0Ch], xmm0           // offset 0x0C: updated angle
```

**stopLevel > 0 路径** (_RDI = result+4, 即 _DWORD* 偏移 1):

```c
*(_RDI - 1) *= 1 << stopLevel;              // offset 0: row *= 2^level
*_RDI *= 1 << stopLevel;                    // offset 4: col *= 2^level
vmulss xmm1, xmm0, dword ptr [rdi+1Ch]     // _RDI+0x1C = offset 0x20: subpixel row
vmulss xmm1, xmm0, dword ptr [rdi+20h]     // _RDI+0x20 = offset 0x24: subpixel col
vcomiss xmm6, dword ptr [rdi+14h]           // _RDI+0x14 = offset 0x18: score (过滤)
```

**关键发现**:
1. row/col 存储为 int32 (像素坐标)，subpixel delta 单独存储在 offset 0x20/0x24
2. 最终浮点坐标 = (float)row + subpixel_row_delta
3. offset 0x14 在输出时从 offset 0x10 复制 (`dest[20]=dest[16]`)，用于非缩放情况下 scaleX=scaleY
4. modelID 在粗搜索阶段写入 offset 0x08

### 8.2 结果截断

```c
*a15 = result_count;

// 如果 v486 (numMatches) > 0 且 < 结果数, 截断
if (v486 > 0 && result_count > v486) {
    *a15 = v486;
}
```

### 8.3 结果排序

结果按 score 降序排列（与单模型相同），不按 modelID 分组。

---

## 9. Phase 7: 后台清理

```c
// beginthreadex 创建线程做资源释放
// 许可证/时间校验 (DRM)
```

与单模型相同的资源清理逻辑。

---

## 10. 常量表

### 10.1 dword_1800D4D50 -- per-level angleStep 倍率表

**地址**: `0x1800D4D50`
**IDA 状态**: 已确认 (与单模型共享)

**值** (float[16]):
```
[0]  = 0.8f
[1]  = 0.9f
[2..15] = 0.9f
```

**用途**: `v532[level] = angleStep_user * dword_1800D4D50[level]`

### 10.2 dword_1800D4CF0 -- per-level angle subdivision 表

**地址**: `0x1800D4CF0`
**IDA 状态**: 已确认 (与单模型共享)

**值** (int[16]):
```
[0]=2, [1]=3, [2]=3, [3]=4, [4]=4, [5]=4, [6]=5,
[7]=5, [8]=5, [9]=5, [10]=6, [11]=6, [12]=6, [13]=6, [14]=6, [15]=7
```

### 10.3 dword_1800D6ADC -- 最粗层 angleStep 因子

**值**: 0.8 (float)
**用途**: 最粗层角度步长公式中的 factor

### 10.4 dword_1800D6B38 -- 非最粗层 angleStep 因子

**值**: 2.0f (float)
**用途**: 非最粗层角度步长公式中的 factor

### 10.5 qword_1800D6BA8 -- rad-to-deg 转换

**值**: 180/PI (= 57.29577951...)
**用途**: 角度步长公式中 acos 结果转 degree

### 10.6 qword_1800D6BA0 -- angleStep 上限

**值**: 11.25 (degree, double)
**用途**: `angleStep_rad = min(qword_1800D6BA0, angleStep_rad)` 上限 clamp

### 10.7 qword_1800D6BD8 / qword_1800D6C00

**值**: 1000.0 (ms 转换, double) / 1e9 (ns 转换, double)
**用途**: 性能监控/计时相关常量

### 10.8 dword_1800D6AA8 -- 初始 cosine (1.0f)

**地址**: `0x1800D6AA8`
**值**: 1.0f (float)
**IDA**: `vmovss xmm6, cs:dword_1800D6AA8`
**用途**: 角度迭代的初始 cos(0) = 1.0，用于搜索起点角度

### 10.9 dword_1800D6BE0 -- 搜索半径 padding

**地址**: `0x1800D6BE0`
**值**: 5.0f (float)
**IDA**: `vaddss xmm1, xmm0, cs:dword_1800D6BE0`
**用途**: 搜索范围计算时添加的边距 padding

### 10.10 dword_1800D6C18 -- 2π (角度回绕)

**地址**: `0x1800D6C18`
**值**: 6.2831853f (2π, float)
**IDA**: `vmovss xmm8, cs:dword_1800D6C18`
**用途**: 角度回绕 (wrap-around)，确保角度在 [0, 2π) 范围内

### 10.11 dword_1800D6CB0 -- -2π (角度回绕)

**地址**: `0x1800D6CB0`
**值**: -6.2831853f (-2π, float)
**IDA**: `vmovss xmm10, cs:dword_1800D6CB0`
**用途**: 角度回绕的负方向，与 dword_1800D6C18 配对使用

### 10.12 dword_1800D6CAC -- 角度范围下界

**地址**: `0x1800D6CAC`
**值**: -180.0f (float, 角度下界 degree)
**IDA**: `vmovss xmm12, cs:dword_1800D6CAC`
**用途**: 角度范围下界 (-180 degree)，与 dword_1800D6C10 / dword_1800D6C08 配合用于角度范围/转换处理

### 10.13 dword_1800D6C10 -- 角度范围上界

**地址**: `0x1800D6C10`
**值**: 180.0f (float, 角度上界 degree)
**IDA**: `vmovss xmm11, cs:dword_1800D6C10`
**用途**: 角度范围上界 (+180 degree)，与 dword_1800D6CAC 配对定义角度合法范围

### 10.14 dword_1800D6C08 -- rad-to-deg 转换 (float)

**地址**: `0x1800D6C08`
**值**: 57.2957801819f (float, 约 180/pi 的 float 精度版)
**IDA**: `vmovss xmm14, cs:dword_1800D6C08`
**用途**: 弧度转角度 (rad-to-deg) 的 float 精度版，与 dword_1800D6CAC / dword_1800D6C10 配合用于角度范围检查和转换

### 10.15 qword_1800D6B00 -- 0.5 (half)

**地址**: `0x1800D6B00`
**值**: 0.5 (double)
**IDA**: `vmovsd xmm13, cs:qword_1800D6B00`
**用途**: 四舍五入 / 坐标偏移半像素

### 10.16 qword_1800D6BF0 -- 秒/天 (许可证时间检查)

**地址**: `0x1800D6BF0`
**值**: 86400.0 (double, 秒/天)
**IDA**: `vdivsd xmm7, xmm0, cs:qword_1800D6BF0`
**用途**: 许可证时间检查，将秒数转换为天数 (seconds / 86400 = days)

### 10.17 xmmword_1800D6E90 -- 符号翻转掩码

**地址**: `0x1800D6E90`
**值**: {0x80000000, 0x80000000, 0x80000000, 0x80000000} (IEEE 754 sign-bit mask, 4x32-bit)
**IDA**: `vmovss xmm15, dword ptr cs:xmmword_1800D6E90`
**用途**: 浮点数符号翻转 (XOR sign bit)，用于角度取反等操作

### 10.18 非算法常量 (许可证/调试)

| 地址 | 类型 | 用途 |
|------|------|------|
| `dword_180108D98` | int | 调试/功能开关 (`if (dword_180108D98 > 0)`) |
| `byte_180108E70` | byte[] | 许可证字节数组 |
| `qword_1800F8120`–`qword_1800F8148` | string ptrs | 许可证字符串数据 |
| `xmmword_1800F8190` | XMM | 许可证时间范围检查 |

> 这些常量与算法无关，仅用于许可证验证和调试输出，QiVision 不需要实现。

---

## 11. 子函数索引

### 11.1 核心搜索子函数 (与单模型共享)

| 地址 | 推测功能 | 单模型对应 | QiVision 对应 | 状态 |
|------|---------|-----------|--------------|------|
| sub_1800B7FB0 | GenerateAngleRange | 同 | GenerateUniformRange | 已对齐 |
| sub_1800B7E20 | 生成缩放搜索网格 | 同 | (缩放路径) | 已对齐 |
| sub_180039480 | BuildResponseMap | 同 | CoarseSearch | 已对齐 |
| sub_1800497F0 | CollectCandidatesNMS | 同 | CollectCandidatesNMS | 已对齐 |
| sub_18004C8C0 | GreedyNMS (最终层) | 同 | NonMaxSuppressionOverlap | 已对齐 |
| sub_18004A5A0 | IntermediateNMS | 同 | (中间层 NMS) | 已对齐 |
| sub_180037E10 | ComputeGradientWithBorder | 同 | (Sobel + 扩展边界) | 已对齐 |
| sub_180036DC0 | ComputeGradient | 同 | (标准 Sobel) | 已对齐 |
| sub_18004B100 | SpatialNMSCluster | 同 | SpatialNMSCluster | 已对齐 |
| sub_18005B950 | SubPixelMode2Worker (Bresenham) | 同 | SubPixelRefine | 已对齐 |
| sub_18005BE10 | SubPixelMode2 LS 拟合 | 同 | SubPixelRefine | 已对齐 |

### 11.2 多模型特有子函数

| 地址 | 推测功能 | 说明 |
|------|---------|------|
| sub_1800B9BE0 | 验证模型句柄有效 | 遍历所有模型 ID 验证 |
| sub_1800806A0 | 模型查找 (by ID) | 从全局红黑树查找模型对象 |
| sub_18007CF60 | 初始化模型数组 | 分配逐模型数据结构 |
| sub_18007E2F0 | 提取/缓存每层特征 | 共享金字塔的特征提取 |
| sub_180067920 | 单任务粗搜索 | 角度候选数 ≤1 时使用。内部: sub_1800B68B0→sub_180039480→sub_1800497F0 |
| sub_18008D9A0 | PPL 并行粗搜索 | 角度候选数 >1 时使用 Concurrency::parallel_for |
| sub_180067BA0 | 单任务细化 | 候选数 ≤1 时使用 |
| sub_18008DD40 | PPL 并行细化 | 候选数 >1 时跨模型统一并行 |
| sub_18004D9C0 | stopLevel>0 缩放细化 | 最终层缩放处理 |
| sub_18004D280 | stopLevel>0 非缩放细化 | 最终层非缩放处理 |
| sub_1800073E0 | 角度搜索网格生成 | 初始化 worker |
| sub_180007930 | 变换操作 | 初始化 worker |
| sub_18007C880 | 缩放路径变换操作 | 细化循环中使用 |
| sub_1800B54A0 | 缩放路径变换操作 | 细化循环中使用 |
| sub_1800C63C0 | 数组最小值 | 辅助函数 |

### 11.3 SubPixel 子函数

| 地址 | 模式 | 缩放 | 说明 |
|------|------|------|------|
| sub_180068030 | mode 1 (parabolic) | 无缩放 | 与单模型 sub_18005B7E0 等效 |
| sub_180068170 | mode 1 (parabolic) | 有缩放 | 缩放版 parabolic |
| sub_18005B950 | mode 2 (LS) | -- | Bresenham 候选评估 |
| sub_18005BE10 | mode 2 (LS) | -- | cv::solve 最小二乘 |

### 11.4 SubPixel 并行子函数

从完整 IDA 代码确认，以下为 PPL parallel_for 包装版本:

| 地址 | 功能 | 对应单任务版 | 说明 |
|------|------|-------------|------|
| sub_18008E0E0 | PPL 并行 SubPixel mode 1 非缩放 | sub_180068030 | parallel_for 包装 |
| sub_18008E480 | PPL 并行 SubPixel mode 1 缩放 | sub_180068170 | parallel_for 包装 |
| sub_18008E820 | PPL 并行 SubPixel mode 2 候选评估 | sub_18005B950 | parallel_for 包装 |
| sub_18008EBC0 | PPL 并行 SubPixel mode 2 LS 拟合 | sub_18005BE10 | parallel_for 包装 |

### 11.5 辅助子函数 (无需实现)

| 地址 | 功能 | 说明 |
|------|------|------|
| sub_18007DFF0 | 初始化搜索范围参数 | 内部初始化 |
| sub_18007FD70 | vector<Result40>::insert | STL 容器操作 |
| sub_1800B9C20 | 全局互斥结果注册 | 线程安全结果收集 |
| sub_1800B7780 | Bresenham 搜索位置生成 | SubPixel mode 2 辅助 |
| sub_180080B90 | 结果数组 copy/overwrite | 内存操作 |
| sub_1800831E0 | 异步清理线程 | beginthreadex 回调 |
| sub_1800B5890 | 计时器获取 | 性能监控 |
| sub_1800B84E0 | 许可证日期解析 | 许可证检查 |

---

## 12. 与 find_shape_model (单数) 的完整对比

| 特性 | find_shape_model | find_shape_models | 差异程度 |
|------|-----------------|-------------------|---------|
| 模型数 | 1 | N (a5) | **核心差异** |
| 金字塔 | 每次调用构建 | **共享，构建一次** | **核心差异** |
| 金字塔层数 | 模型自身 numLevels | max(所有模型 numLevels) | 差异 |
| level 范围 | 全局参数 | **逐模型 (a12 数组)** | **核心差异** |
| 并行策略 | 单模型内并行 | **跨模型候选统一并行 (PPL)** | **核心差异** |
| 粗搜索 | 对单模型搜索 | **逐模型顺序粗搜, 结果合并** | 差异 |
| 细化 | 单模型候选细化 | **混合候选统一 parallel_for** | 差异 |
| 结果格式 | (row, col, angle, score) | **(row, col, modelID, angle, score, ...)** | **核心差异** |
| 结果排序 | score 降序 | score 降序 (不按 modelID 分组) | 相同 |
| SubPixel | mode 0/1/2/3 | mode 0/1/2/3 (同) | 相同 |
| 响应图 LUT | 同 | 同 | 相同 |
| NMS 后处理 | SpatialNMSCluster | SpatialNMSCluster (同) | 相同 |
| 核心子函数 | sub_18004C8C0 等 | **复用相同子函数** | 相同 |
| 常量表 | dword_1800D4D50/CF0 | 同 | 相同 |
| 缩放支持 | 否 (find_shape_model) | **有** (v520 > 0 缩放路径, 已确认) | **差异** |
| mask 支持 | 否 | 否 (find_shape_models_2 有) | 相同 |

---

## 13. 整体流程架构

```
find_shape_models 流程:
  |
  +-- Phase 0: 初始化
  |   +-- 验证输入 (a1 != null, 模型句柄有效)
  |   +-- 构建 cv::Mat 包装图像
  |   +-- 逐模型提取属性 (numLevels, 各 offset 值)
  |   +-- 计算全局 maxNumLevels, minStartLevel
  |   +-- 解析 a12 逐模型 (numLevels, startLevel) 对
  |
  +-- Phase 1: 共享金字塔构建 (一次构建, 所有模型共享)
  |   +-- for i in 0..maxNumLevels-1:
  |       +-- i > 0: cv::pyrDown
  |       +-- sub_18007E2F0: 提取/缓存每层特征
  |
  +-- Phase 2: angleStep 表填充
  |   +-- v532[level] = angleStep * dword_1800D4D50[level]
  |   +-- 最粗层 angleStep 特殊公式 (acos + factor)
  |
  +-- Phase 3: 逐模型最粗层搜索
  |   +-- for v512 in 0..modelCount-1:
  |   |   +-- 找到当前模型最粗有效层 v509
  |   |   +-- 初始化搜索参数 (if needed)
  |   |   +-- sub_1800B7FB0: 生成角度搜索网格
  |   |   +-- sub_1800B7FB0: 生成搜索候选位置
  |   |   +-- PPL 并行搜索:
  |   |   |   +-- sub_180067920 (单任务) 或 sub_18008D9A0 (多任务)
  |   |   +-- 结果标记 modelID: v170[v172+8] = a4[v512]
  |   +-- 所有模型的粗搜结果合并到统一候选池
  |
  +-- Phase 4: 金字塔细化 (从粗到细)
  |   +-- for v509 from v510-2 downto v511:
  |       +-- 收集上一层候选 (混合所有模型)
  |       +-- if v520 > 0 (有缩放):
  |       |   +-- sub_1800B7E20: 生成缩放搜索网格
  |       |   +-- sub_18007C880 + sub_1800B54A0: 变换
  |       +-- 准备本层数据:
  |       |   +-- sub_180036DC0 (有缩放) / sub_180037E10 (无缩放)
  |       +-- PPL 并行细化 (跨模型统一):
  |       |   +-- sub_180067BA0 (单任务) 或 sub_18008DD40 (多任务)
  |       +-- 角度范围调整 (wrap-around)
  |       +-- 候选过滤: score < v532[level] 移除
  |       +-- 最终层 (v509 == v511):
  |           +-- if v520: sub_18004D9C0 (缩放细化)
  |           +-- else:    sub_18004D280 (非缩放细化, greediness)
  |
  +-- Phase 4b: SubPixel 处理
  |   +-- mode 1: sub_180068030/sub_180068170 (parabolic)
  |   +-- mode 2: sub_18005B950 + sub_18005BE10 (LS, 两次迭代)
  |
  +-- Phase 5: stopLevel 输出缩放
  |   +-- if v511 > 0:
  |       +-- 坐标 *= 2^stopLevel
  |       +-- score < minScore 过滤
  |
  +-- Phase 6: 结果输出
  |   +-- 拷贝到 a14 (每结果 40 字节, 含 modelID)
  |   +-- *a15 = result_count
  |   +-- numMatches > 0 时截断
  |
  +-- Phase 7: 后台清理
      +-- beginthreadex 资源释放
```

---

## 14. QiVision 实现策略

### 14.1 当前实现状态

| 功能 | 状态 | 位置 |
|------|------|------|
| FindShapeModel (单模型, 非缩放) | DONE | ShapeModelSearch.cpp |
| FindScaledShapeModel (单模型, 缩放) | DONE | ShapeModelSearch.cpp |
| FindShapeModel + mask | DONE | DownsampleMask + MaskGradientLevel |
| **FindShapeModels (多模型)** | **未实现** | -- |

### 14.2 需要新增的功能

| # | 功能 | 优先级 | 工作量 | 说明 |
|---|------|--------|--------|------|
| 1 | 公开 API: `FindShapeModels` | P0 | 小 | 新增 API 函数签名 |
| 2 | 共享金字塔构建 | P0 | 中 | 现有 AnglePyramid 按单模型设计，需扩展为取 max(numLevels) |
| 3 | 逐模型 level 参数 | P0 | 小 | a12 对应的逐模型 (numLevels, startLevel) |
| 4 | 多模型粗搜索循环 | P0 | 中 | 逐模型粗搜 + 结果合并 + modelID 标记 |
| 5 | 跨模型候选统一细化 | P0 | 中 | 混合候选池 + 按 modelID 分派模板 |
| 6 | 结果格式扩展 (含 modelID) | P0 | 小 | ShapeMatchResult 增加 modelIndex 字段 |
| 7 | PPL/OpenMP 并行策略 | P1 | 中 | 跨模型候选的负载均衡 |
| 8 | stopLevel 逐模型不同 | P1 | 小 | 每个模型可能有不同的 startLevel |

### 14.3 可复用的现有实现

以下部分已在 QiVision 中实现，可直接复用:

- **金字塔构建**: `Internal::AnglePyramid` (需参数化层数)
- **粗搜索**: `CoarseSearch()` (逐模型调用，合并结果)
- **角度/缩放范围生成**: `GenerateUniformRange()`
- **细化管线**: `RefineAtLevel()` / `RefineAtLevelScaled()` (需传入正确模型)
- **SubPixel**: `SubPixelRefine()` (完全复用)
- **NMS**: `SpatialNMSCluster()` / `NonMaxSuppressionOverlap()` (完全复用)
- **结果排序截断**: 完全复用

### 14.4 推荐 API 设计

```cpp
// Halcon 风格 API
QIVISION_API void FindShapeModels(
    const QImage& image,
    const std::vector<ShapeModel>& models,    // 多个模型
    double angleStart,
    double angleExtent,
    double minScore,
    int32_t numMatches,
    double maxOverlap,
    const std::string& subPixel,
    const std::vector<int32_t>& numLevels,    // 逐模型层数 (0=auto)
    double greediness,
    std::vector<double>& rows,
    std::vector<double>& cols,
    std::vector<double>& angles,
    std::vector<double>& scores,
    std::vector<int32_t>& modelIndices,       // 输出: 每个匹配的模型索引
    const QImage& searchMask = QImage()
);
```

### 14.5 实现架构建议

```
FindShapeModels():
  |
  +-- 1. 验证: 所有模型有效, 图像非空
  +-- 2. 计算全局参数:
  |       maxNumLevels = max(model[i].numLevels for all i)
  |       minStartLevel = min(startLevel[i] for all i)
  +-- 3. 共享金字塔: BuildPyramid(image, maxNumLevels)
  +-- 4. 逐模型粗搜索:
  |       for each model:
  |           candidates[model] = CoarseSearch(pyramid[coarsest], model)
  |           tag each candidate with modelIndex
  +-- 5. 合并候选池: allCandidates = merge(candidates[0..N-1])
  +-- 6. 金字塔细化 (共享):
  |       for level from coarsest-1 downto startLevel:
  |           for each candidate in allCandidates:
  |               refine using candidate.modelIndex -> select model template
  |           filter by levelThreshold
  +-- 7. SubPixel + NMS + sort + truncate
  +-- 8. 输出 (含 modelIndices)
```

### 14.6 关键实现细节

| 细节 | 反编译行为 | 实现策略 |
|------|-----------|---------|
| 候选 modelID 标记 | v170[v172+8] = a4[v512] | 在 Candidate 结构体增加 modelIndex 字段 |
| 跨模型细化 | 混合候选在同一 parallel_for | 每个候选携带 modelIndex，细化时按 index 选模板 |
| 金字塔共享 | 单次 pyrDown 循环 | BuildPyramid 取 max 层数，所有模型共享 |
| 逐模型 level 参数 | a12 数组 | vector<int32_t> numLevels 参数 |
| 结果排序 | 全局 score 降序 | 不按 modelIndex 分组，统一排序 |

### 14.7 性能优势分析

| 场景 | 逐个调用 FindShapeModel | FindShapeModels |
|------|------------------------|-----------------|
| N 个模型 | N 次金字塔构建 + N 次 Sobel | **1 次金字塔 + 1 次 Sobel** |
| 图像预处理 | O(N * imageSize) | O(imageSize) |
| 并行粒度 | 模型内候选并行 | **跨模型候选统一并行** |
| 典型加速比 | 1x | **约 N× (金字塔共享) + 并行均衡** |

---

## 15. 待确认事项

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| ~~1~~ | ~~a6-a10 的精确参数映射~~ | **已确认** (2026-03-11) | 完整参数映射见 §0.1 和 §0.3 |
| ~~2~~ | ~~v520 缩放路径~~ | **已确认** (2026-03-11) | find_shape_models 内含缩放路径 (v520 > 0)，不需要单独的 find_scaled_shape_models |
| ~~5~~ | ~~sub_180067920 / sub_18008D9A0 的候选阈值~~ | **已确认** (2026-03-11) | 阈值 = 1: `if (count > 1)` 并行, 否则单任务 |
| ~~6~~ | ~~结果 40 字节完整格式~~ | **已确认** (2026-03-11) | 完整 10 字段结构见 §8.1 |
| ~~7~~ | ~~find_scaled_shape_models 是否存在~~ | **已确认** (2026-03-11) | 不存在，缩放路径内含于 find_shape_models (v520 判断) |

### 已确认详情 (2026-03-11)

**参数映射**: 基于完整 IDA 反编译代码确认所有 15 个参数的精确类型、寄存器/栈位置和语义。关键发现: a6=AngleExtent (xmm8), a7=AngleStart (xmm7)，顺序与 Halcon 文档相反。

**40 字节结果格式**: 从 SubPixel mode 2 的读写操作 (r14 偏移) 和 stopLevel>0 的坐标缩放代码确认全部 10 个字段。关键发现: row/col 为 int32 (非 float)，subpixel delta 单独存储在 offset 0x20/0x24；offset 0x14 在输出时从 0x10 复制。

**缩放路径**: `v520 = *(_DWORD *)(v122[12] + 4 * _RDX)` 从模型读取缩放标志。当 `v520 > 0` 时走缩放分支 (`sub_1800B7E20` 生成缩放网格, `sub_18004D9C0` 缩放细化)，否则走非缩放分支 (`sub_18004D280`)。这意味着 find_shape_models 统一处理缩放和非缩放模型。

> **已确认常量** (2026-03-10): dword_1800D6B38=2.0f, qword_1800D6BA0=11.25, dword_1800D6BE0=5.0f, dword_1800D6CAC=-180.0f, dword_1800D6C10=180.0f, dword_1800D6C08=57.2958f, xmmword_1800D6E90=0x80000000x4, qword_1800D6BF0=86400.0
