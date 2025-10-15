(function () {
  function ready(fn){document.readyState!=='loading'?fn():document.addEventListener('DOMContentLoaded',fn);}
  ready(function(){
    var side = document.querySelector('.wy-nav-side');
    if(!side) return;

    // ★ 这里写你的扉页绝对/相对路径
    // 如果扉页就在站点根目录：site_root/index.html
    var target = 'D:/Polyu/AI%20based%20solvers/website/site/index.html'; //需要改成路径

    var btn = document.createElement('a');
    btn.className = 'hpr-back-fab';
    btn.textContent = 'Back to HPR homepage';
    btn.href = target;   // ⬅️ 直接跳扉页

    document.body.appendChild(btn);

    function syncPos(){
      var r = side.getBoundingClientRect();
      btn.style.left  = r.left + 'px';
      btn.style.width = r.width + 'px';
    }
    syncPos();
    window.addEventListener('resize', syncPos);
    window.addEventListener('scroll',  syncPos);
  });
})();
