'use strict';
const methods={
 data:{label:'DATA PREPARATION',title:'先统一任务，再组织数据。',body:'围绕交通标志异常识别完成样本去重、标签映射与多模态指令构建，统一任务描述和输出格式，让训练样本与评测目标保持一致。',items:[['样本处理','去重与标签映射'],['任务描述','图像 + 异常识别指令'],['目标输出','结构化 JSON 标签']]},
 sft:{label:'SUPERVISED FINE-TUNING',title:'低比特基座，轻量参数更新。',body:'基于 NF4 量化与 QLoRA，对 Qwen3-VL 的 Q / K / V / O 注意力投影进行参数高效微调，将多模态理解能力适配到交通标志异常识别与结构化输出。',items:[['基座表示','NF4 低比特量化'],['训练方式','LoRA 低秩适配'],['任务目标','异常标签与输出格式']]},
 rl:{label:'POST-TRAINING RESEARCH',title:'让任务评价参与策略优化。',body:'研究基于 GRPO 的后训练方案，围绕识别正确性、多标签覆盖与输出约束设计可验证奖励，通过对比和消融评测分析奖励组成的作用。',items:[['优化思路','组内相对奖励'],['奖励来源','标签与格式校验'],['研究重点','奖励设计与消融评测']]}
};
const rewards={
 correctness:{label:'SEMANTIC ACCURACY',title:'异常类别是否判断正确？',body:'将候选输出中的异常类别与标注进行比较，为正确识别提供奖励信号。评价依据是任务标签，而不是回答的长度。',formula:'预测类别 ↔ 标注类别',detail:'关注视觉语义是否正确，避免仅生成格式规范但内容错误的答案。'},
 f1:{label:'MULTI-LABEL COVERAGE',title:'识别完整，也减少误报。',body:'同一张图像可能包含多种异常。多标签 F1 同时考虑 Precision 与 Recall，兼顾标签覆盖和预测准确性。',formula:'F1 = 2 × P × R / (P + R)',detail:'需要统一标签集合，并约定空标签集合与零分母情形的处理方式。'},
 format:{label:'OUTPUT VALIDITY',title:'结果可以直接被程序解析。',body:'检查输出是否为合法 JSON、必要字段是否存在，以及字段类型是否符合任务约定，让识别结果能够进入后续处理流程。',formula:'JSON 解析 → 字段检查 → 类型校验',detail:'格式奖励约束输出结构；它本身不证明异常识别正确。'},
 consistency:{label:'LABEL CONSISTENCY',title:'让不同字段表达同一结论。',body:'检查异常标记与类别列表是否一致、标签是否属于合法集合，以及输出中是否出现互相矛盾的字段。',formula:'异常标记 ↔ 标签列表 ↔ 标签集合',detail:'例如：异常标记为 false 时，不应同时给出非空异常类别。'}
};
const examples={
 stain:{n:'01',title:'污损 · 表面信息被遮挡',desc:'关注泥污、涂抹等局部覆盖，以及覆盖对标志内容辨识的影响。',label:'污损'},
 deform:{n:'02',title:'变形 · 结构与轮廓变化',desc:'关注标志牌的弯折、翘曲与外形变化，区分结构异常和拍摄视角造成的差异。',label:'变形'},
 age:{n:'03',title:'老化 · 外观与材质退化',desc:'关注褪色、锈蚀与表面剥落等线索，并结合光照条件理解外观变化。',label:'老化'},
 break:{n:'04',title:'破坏 · 标志完整性受损',desc:'关注裂口、断裂或局部缺损，识别标志牌完整性发生变化的区域。',label:'破坏'},
 missing:{n:'05',title:'脱落 · 连接与安装状态异常',desc:'关注标志牌脱离支撑结构、悬挂或倒落等状态，并结合场景上下文进行判断。',label:'脱落'}
};
function connectTabs(selector,update){
 const buttons=[...document.querySelectorAll(selector)];
 function activate(button){buttons.forEach(b=>{b.setAttribute('aria-selected',String(b===button));b.tabIndex=b===button?0:-1;});update(button);}
 buttons.forEach((button,index)=>{button.addEventListener('click',()=>activate(button));button.addEventListener('keydown',event=>{let i;if(event.key==='ArrowRight'||event.key==='ArrowDown')i=(index+1)%buttons.length;else if(event.key==='ArrowLeft'||event.key==='ArrowUp')i=(index-1+buttons.length)%buttons.length;else if(event.key==='Home')i=0;else if(event.key==='End')i=buttons.length-1;else return;event.preventDefault();activate(buttons[i]);buttons[i].focus();});});activate(buttons[0]);
}
connectTabs('[data-method]',b=>{const d=methods[b.dataset.method];const p=document.getElementById('method-panel');p.setAttribute('aria-labelledby',b.id);p.innerHTML=`<div class="eyebrow">${d.label}</div><h3>${d.title}</h3><p>${d.body}</p><dl>${d.items.map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>`;});
connectTabs('[data-reward]',b=>{const d=rewards[b.dataset.reward];const p=document.getElementById('reward-panel');p.setAttribute('aria-labelledby',b.id);p.innerHTML=`<div class="eyebrow">${d.label}</div><h3>${d.title}</h3><p>${d.body}</p><div class="formula">${d.formula}</div><p class="reward-detail">${d.detail}</p>`;});
function signArt(key){
 const base='<circle cx="230" cy="132" r="78" fill="#e3e8d9"/><circle cx="230" cy="132" r="67" fill="none" stroke="#ab6250" stroke-width="12"/><text x="230" y="155" text-anchor="middle" font-family="Arial,sans-serif" font-size="65" font-weight="700" fill="#303e31">60</text>';
 const defects={stain:'<path d="M248 83l26 7 13 28-22 23-14-13-28 6-6-21 19-8z" fill="#74745b"/><circle cx="279" cy="152" r="8" fill="#74745b"/>',deform:'<path d="M260 64l-14 63 22 76" fill="none" stroke="#51624b" stroke-width="8" opacity=".55"/>',age:'<path d="M183 84l12 10m64-8l-5 8m38 37l-9 4m-91 39l11-7m42 18l8-13" stroke="#8b6b42" stroke-width="7"/><path d="M179 105l92 56m-88-23l67 37" stroke="#d1d9bf" stroke-width="6" opacity=".6"/>',break:'<path d="M230 54l-9 49 22 26-20 28 14 53" fill="none" stroke="#1d261f" stroke-width="10"/><path d="M290 91l19 20-7 34-20-18z" fill="#1d261f"/>',missing:''};
 const transform=key==='missing'?'translate(110 265) rotate(-71) translate(-230 -132)':key==='deform'?'translate(53 0) skewX(-13)':'rotate(-5 230 132)';
 return `<path d="M35 294h390" stroke="#4c6044"/><path d="M35 306h390" stroke="#2d3d28"/><rect x="223" y="167" width="14" height="127" rx="2" fill="#6d7b66"/>${key==='missing'?'<path d="M220 112h20m-20 12h20" stroke="#9dac91" stroke-width="4"/>':''}<g transform="${transform}" ${key==='age'?'opacity=".65"':''}>${base}${defects[key]}</g><path d="M73 54V36h18M369 36h18v18M387 264v18h-18M91 282H73v-18" fill="none" stroke="#8ba97a" stroke-width="1"/>`;
}
connectTabs('[data-example]',b=>{const key=b.dataset.example,d=examples[key];document.getElementById('example-panel').setAttribute('aria-labelledby',b.id);document.getElementById('example-number').textContent='ANOMALY / '+d.n;document.getElementById('example-title').textContent=d.title;document.getElementById('example-description').textContent=d.desc;document.getElementById('example-json').textContent=JSON.stringify({has_anomaly:true,anomaly_labels:[d.label]},null,2);const svg=document.getElementById('example-art');svg.setAttribute('aria-label','交通标志'+d.label+'示意');svg.innerHTML=signArt(key);});
