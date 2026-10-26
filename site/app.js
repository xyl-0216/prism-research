// All content is a local historical case replay. No inference or network requests.
const sourceURL = 'https://www.nvidia.com/en-us/geforce/news/rtx-5070-out-now/';
const panels = [
  {title:'图例、脚注与明确数值', rows:[
    ['比较对象','RTX 5070 / RTX 4070','图例 · 视觉转录','绿色对应 RTX 5070，灰色对应 RTX 4070。','确认同一组比较对象。'],
    ['游戏脚注','1440p；两代显卡使用不同帧生成设置','脚注 · 视觉转录','40 系列：Frame Gen；50 系列：Multi Frame Gen（4X）。','图中不是关闭帧生成的配对测试。'],
    ['明确标注','Flux.dev：3.7X','图中标签 · 视觉转录','3.7X 对应生成式 AI 工作负载，精度条件为 FP4。','该数值不属于游戏 FPS。']
  ]},
  {title:'追溯到原始发布文章', rows:[
    ['来源','NVIDIA 官方发布文章','原始网页','RTX 5070 发布文章与配套性能图。','保留发布方身份与来源链接。'],
    ['正文描述','传统光栅性能提升 20%','厂商正文','原文描述：20% boost in traditional raster performance。','这是厂商正文描述，不是由图中柱高估算的数值。'],
    ['追溯记录','网页与图片分别归档','本地运行记录','页面、图片及视觉观察记录关联来源 URL 与内容哈希。','回到原始资料检查结论依据。']
  ]},
  {title:'同一结论，需要匹配的条件', rows:[
    ['帧生成','要求关闭；图中启用 → 条件不匹配','关联证据 · 游戏脚注','Frame Gen / Multi Frame Gen（4X）。','不能据此推导关闭帧生成时的表现。'],
    ['指标','AI 吞吐量 ≠ 游戏帧率','关联证据 · Flux.dev 标签','3.7X 属于生成式 AI 测试。','不能将 AI 指标用作游戏帧率证据。'],
    ['范围','所列测试项目 ≠ 所有游戏','关联证据 · 测试范围','现有图表只展示所列项目。','不能直接外推为所有游戏均翻倍。']
  ]},
  {title:'现有证据支持什么？', verdict:'证据不足，不能证明该主张。', rows:[
    ['关键依据','测试条件与主张不一致','结论关联 · 条件核验','图中启用帧生成，问题要求关闭帧生成。','缺少条件匹配的逐游戏数据。'],
    ['结论范围','不能证明，不等于证明处处为假','结论关联 · 证据范围','现有材料未覆盖所有游戏和所要求的测试设置。','保留未确定项，不把证据不足当成全面反驳。']
  ]}
];
const tabs=[...document.querySelectorAll('[data-step]')];
const panel=document.getElementById('evidence-panel');
const pauseButton=document.getElementById('replay-pause');
const status=document.getElementById('replay-status');
const progress=document.getElementById('replay-progress');
const motion=window.matchMedia('(prefers-reduced-motion: reduce)');
let index=0, playing=false, timer=null, completed=false, visible=false, userStopped=false;
const escapeHTML=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function updateControls(){
  pauseButton.hidden=!playing;
  status.textContent=`${completed?'浏览完成':playing?'自动浏览':'手动浏览'} · ${index+1} / ${panels.length}`;
  progress.value=index+1;
}
function pause(){clearTimeout(timer);timer=null;playing=false;updateControls()}
function stop(){userStopped=true;pause()}
function autoBrowse(){
  if(!visible || document.hidden || motion.matches || userStopped || completed || playing)return;
  playing=true;updateControls();schedule();
}
function highlight(rowIndex){
  const row=panels[index].rows[rowIndex];
  panel.querySelectorAll('[data-evidence]').forEach((button,i)=>{
    button.classList.toggle('linked',i===rowIndex);
    button.setAttribute('aria-pressed',String(i===rowIndex));
  });
  const detail=document.getElementById('source-detail');
  detail.innerHTML=`<span class="source-label">${escapeHTML(row[2])}</span><p>${escapeHTML(row[3])}</p><div class="source-inference">${escapeHTML(row[4])}</div><a href="${sourceURL}" target="_blank" rel="noopener">查看原始来源 ↗</a>`;
  if(!motion.matches && detail.animate)detail.animate([{opacity:.45},{opacity:1}],{duration:180});
}
function render(focus=false){
  tabs.forEach((tab,i)=>{tab.classList.toggle('active',i===index);tab.classList.toggle('visited',i<index);tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1});
  const data=panels[index];
  panel.innerHTML=`<span class="label">步骤 ${index+1} / 历史证据</span><h3>${data.title}</h3>${data.verdict?`<div class="verdict">${data.verdict}</div>`:''}<div class="evidence-links">${data.rows.map((r,i)=>`<button type="button" class="evidence-row" data-evidence="${i}" aria-pressed="false" aria-controls="source-detail"><span>${r[0]}</span><strong>${r[1]}</strong><b aria-hidden="true">↗</b></button>`).join('')}</div><aside id="source-detail" class="source-detail" aria-label="关联证据" aria-live="polite"></aside>`;
  panel.setAttribute('aria-labelledby',`tab-${index}`);
  panel.querySelectorAll('[data-evidence]').forEach((button,i)=>button.addEventListener('click',()=>{stop();highlight(i)}));
  highlight(0);
  if(!motion.matches && panel.animate)panel.animate([{opacity:.45,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:230,easing:'ease-out'});
  if(focus)tabs[index].focus();
  updateControls();
}
function schedule(){
  clearTimeout(timer);
  timer=setTimeout(()=>{
    if(!playing)return;
    if(index===panels.length-1){completed=true;pause();return}
    index++;render();schedule();
  },6000);
}
pauseButton.addEventListener('click',stop);
tabs.forEach((tab,i)=>{
  const choose=focus=>{stop();index=i;completed=false;render(focus)};
  tab.addEventListener('click',()=>choose(false));
  tab.addEventListener('keydown',event=>{
    let next;
    if(['ArrowRight','ArrowDown'].includes(event.key))next=(i+1)%4;
    if(['ArrowLeft','ArrowUp'].includes(event.key))next=(i+3)%4;
    if(event.key==='Home')next=0;if(event.key==='End')next=3;
    if(next!==undefined){event.preventDefault();stop();index=next;completed=false;render(true)}
  });
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();else autoBrowse()});
motion.addEventListener('change',()=>{if(motion.matches){pause();document.getAnimations?.().forEach(a=>a.cancel());document.querySelectorAll('.reveal-pending').forEach(el=>el.classList.remove('reveal-pending'))}else autoBrowse()});
if('IntersectionObserver' in window){
  const visibility=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting && entries[0].intersectionRatio>=.2;if(visible)autoBrowse();else pause()},{threshold:[0,.2]});
  visibility.observe(document.querySelector('.case-shell'));
  if(!motion.matches){
    const reveal=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.remove('reveal-pending');reveal.unobserve(entry.target)}}),{threshold:.08});
    document.querySelectorAll('#system, #research, #release').forEach(el=>{el.classList.add('reveal-pending');reveal.observe(el)});
  }
}else{visible=true}
render();
autoBrowse();
