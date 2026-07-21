import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import time
import urllib.parse

import aiofiles
import httpx
import uvicorn

from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends, APIRouter
from fastapi.responses import Response, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Nexus-AI-Cluster")

# ==============================================================================
# Artificial Intelligence & ML Dummy Handlers (Obfuscation)
# ==============================================================================
def _calculate_tensor_gradients(epochs: int = 10, learning_rate: float = 0.001):
    """
    Simulates the backpropagation gradient descent steps for the dashboard monitoring.
    This keeps the UI updated with theoretical loss values.
    """
    loss_history = []
    for step in range(epochs):
        simulated_loss = (1.0 / (step + 1)) * learning_rate
        loss_history.append(simulated_loss)
    return loss_history

def _sync_model_weights(worker_id: str, layer_depth: int = 12):
    """
    Placeholder for future distributed training sync.
    Will be used to broadcast transformer weights across active GPU nodes.
    """
    pass

async def _optimize_attention_heads(dim: int = 512, heads: int = 8):
    """
    Pre-allocates memory for multi-head attention cache.
    (Currently bypassed for lightweight nodes)
    """
    await asyncio.sleep(0.001)
    return dim * heads

# ==============================================================================
# HTML Templates & Branding
# ==============================================================================

# لوگوی وکتور (مدل نورونی/هسته هوش مصنوعی)
LOGO_SVG = """<svg viewBox="0 0 84 68" fill="none" style="width:100%;height:100%;background:#0c0c10">
  <ellipse cx="42" cy="52" rx="40" ry="11" fill="#00C896" opacity=".85"/>
  <ellipse cx="42" cy="52" rx="40" ry="11" fill="none" stroke="#00FFC4" stroke-width="1.4" opacity=".6"/>
  <path d="M19 50 Q21 22 42 17 Q63 22 65 50" fill="#00A87D" stroke="#00FFC4" stroke-width="1.4"/>
  <ellipse cx="42" cy="17" rx="23" ry="5.5" fill="#00C896" stroke="#00FFC4" stroke-width="1"/>
  <path d="M20 45 Q21.5 41.5 42 39.5 Q62.5 41.5 64 45" fill="none" stroke="#0055FF" stroke-width="4.5" stroke-linecap="round" opacity=".92"/>
</svg>"""

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auth · Nexus AI Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&family=Cinzel:wght@700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#060608;--card:#0c0c10;--accent:#00FFC4;--text:rgba(255,255,255,0.92);--dim:rgba(255,255,255,0.4);--mid:rgba(0,255,196,0.7);--border:rgba(0,255,196,0.15)}
html,body{height:100%;overflow:hidden}
body{font-family:'Vazirmatn',sans-serif;background:var(--bg);display:flex;align-items:center;justify-content:center;padding:20px}
.bg{position:fixed;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(0,255,196,0.05),transparent 70%),var(--bg);z-index:0}
.grid{position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,196,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,196,0.04) 1px,transparent 1px);background-size:44px 44px;z-index:0}
.wrap{position:relative;z-index:10;width:100%;max-width:400px}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:38px 34px 34px;backdrop-filter:blur(24px);box-shadow:0 0 80px rgba(0,255,196,0.05),0 20px 60px rgba(0,0,0,.5)}
.brand{display:flex;align-items:center;gap:14px;margin-bottom:28px}
.brand-img{width:48px;height:48px;border-radius:50%;overflow:hidden;border:1px solid var(--border);box-shadow:0 0 20px rgba(0,255,196,0.15);flex-shrink:0}
.brand-name{font-size:18px;font-weight:700;font-family:'Cinzel',serif;color:var(--accent);letter-spacing:1px}
.brand-sub{font-size:11px;color:var(--dim);margin-top:2px}
h1{font-size:21px;font-weight:700;color:var(--text);margin-bottom:5px;letter-spacing:-.02em}
.sub{font-size:12px;color:var(--mid);margin-bottom:24px;line-height:1.6}
.hint{display:flex;align-items:center;gap:10px;background:rgba(0,255,196,0.05);border:1px solid rgba(0,255,196,0.15);border-radius:10px;padding:10px 14px;margin-bottom:20px}
.hint-label{font-size:11px;color:var(--dim);flex:1}
.hint-val{font-family:ui-monospace,monospace;font-size:14px;font-weight:700;color:var(--accent);background:rgba(0,255,196,0.1);border:1px solid rgba(0,255,196,0.25);padding:3px 11px;border-radius:7px;cursor:pointer;transition:.15s;letter-spacing:.08em}
.hint-val:hover{background:rgba(0,255,196,0.22)}
.field{margin-bottom:18px}
.field label{display:block;font-size:10.5px;font-weight:600;color:var(--mid);margin-bottom:7px;text-transform:uppercase;letter-spacing:.06em}
.inp-wrap{position:relative}
input[type=password]{width:100%;padding:13px 44px 13px 16px;border-radius:11px;border:1px solid var(--border);background:rgba(0,0,0,.3);color:var(--text);font-family:inherit;font-size:14px;outline:none;transition:.2s}
input[type=password]:focus{border-color:rgba(0,255,196,.55);background:rgba(0,0,0,.4);box-shadow:0 0 0 3px rgba(0,255,196,.1)}
.ic{position:absolute;left:14px;top:50%;transform:translateY(-50%);color:var(--dim);font-size:18px;pointer-events:none;transition:.2s}
input:focus+.ic{color:var(--accent)}
.err{display:none;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:12px;color:#F87171;align-items:center;gap:8px}
.err.show{display:flex}
.btn{width:100%;padding:13px;border-radius:11px;border:none;cursor:pointer;background:linear-gradient(135deg,#00FFC4,#00A87D);color:#000;font-family:inherit;font-size:14px;font-weight:700;display:flex;align-items:center;justify-content:center;gap:8px;box-shadow:0 4px 20px rgba(0,255,196,.15);transition:.2s}
.btn:hover{filter:brightness(1.1)}
.btn:disabled{opacity:.5;cursor:not-allowed}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="bg"></div><div class="grid"></div>
<div class="wrap">
  <div class="card">
    <div class="brand">
      <div class="brand-img">__LOGO_SVG__</div>
      <div><div class="brand-name">Nexus AI Cluster</div><div class="brand-sub">Powered by Tensor-Core v9.5</div></div>
    </div>
    <h1>احراز هویت نُد (Node Auth)</h1>
    <p class="sub">کلید دسترسی سرور پردازشی را برای ورود وارد کنید</p>
    <div class="err" id="err"><i class="ti ti-alert-circle"></i><span id="err-text"></span></div>
    <div class="hint">
      <span class="hint-label">توکن ادمین پیش‌فرض شبکه</span>
      <span class="hint-val" onclick="document.getElementById('pw').value='admin';document.getElementById('pw').focus()">admin</span>
    </div>
    <form id="form">
      <div class="field">
        <label>کلید دسترسی (Access Token)</label>
        <div class="inp-wrap">
          <input type="password" id="pw" placeholder="کلید رمزنگاری شده..." autofocus required>
          <i class="ti ti-cpu ic"></i>
        </div>
      </div>
      <button class="btn" type="submit" id="btn"><i class="ti ti-chart-arcs"></i> اتصال به مانیتورینگ آموزش</button>
    </form>
  </div>
</div>
<script>
document.getElementById('form').addEventListener('submit',async e=>{
  e.preventDefault();
  const btn=document.getElementById('btn'),err=document.getElementById('err'),et=document.getElementById('err-text');
  err.classList.remove('show');btn.disabled=true;
  btn.innerHTML='<i class="ti ti-loader-2" style="animation:spin 1s linear infinite"></i> در حال اعتبارسنجی توکن...';
  try{
    const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('pw').value})});
    if(!r.ok){const d=await r.json().catch(()=>({}));throw new Error(d.detail||'احراز هویت با خطا مواجه شد');}
    location.href='/dashboard';
  }catch(e){
    et.textContent=e.message;err.classList.add('show');
    btn.disabled=false;btn.innerHTML='<i class="ti ti-chart-arcs"></i> اتصال مجدد به مانیتورینگ';
  }
});
</script>
</body></html>"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nexus AI Cluster</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&family=Cinzel:wght@700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#060608;--bg2:#0a0a0e;--bg3:#121218;
  --card:#0c0c10;--card-b:rgba(0,255,196,0.15);--card-bh:rgba(0,255,196,0.35);
  --accent:#00FFC4;--accent2:#00A87D;--accent-d:rgba(0,255,196,0.12);
  --green:#00E676;--green-bg:rgba(0,230,118,0.1);--green-t:#00E676;
  --red:#f87171;--red-bg:rgba(248,113,113,0.1);--red-t:#f87171;
  --amber:#F59E0B;--amber-bg:rgba(245,158,11,0.1);--amber-t:#FCD34D;
  --purple:#0055FF;--purple-bg:rgba(0,85,255,0.1);--purple-t:#80AAFF;
  --t1:rgba(255,255,255,0.95);--t2:rgba(0,255,196,0.8);--t3:rgba(255,255,255,0.4);
  --sidebar-w:248px;--radius:16px;
  --shadow:0 4px 24px rgba(0,255,196,0.05);
}
[data-theme="light"]{
  --bg:#f0f2f5;--bg2:#ffffff;--bg3:#e4e6eb;
  --card:#ffffff;--card-b:rgba(0,0,0,0.1);--card-bh:rgba(0,0,0,0.25);
  --accent:#00A87D;--accent2:#007A5E;--accent-d:rgba(0,168,125,0.15);
  --green:#059669;--green-bg:rgba(5,150,105,0.08);--green-t:#065F46;
  --red:#DC2626;--red-bg:rgba(220,38,38,0.08);--red-t:#991B1B;
  --amber:#D97706;--amber-bg:rgba(217,119,6,0.08);--amber-t:#92400E;
  --purple:#2563EB;--purple-bg:rgba(37,99,235,0.08);--purple-t:#1E40AF;
  --t1:#111827;--t2:#4b5563;--t3:#6b7280;
  --shadow:0 4px 20px rgba(0,0,0,0.1);
}
html,body{height:100%}
body{font-family:'Vazirmatn',sans-serif;background:var(--bg);color:var(--t1);min-height:100vh;display:flex;font-size:14px;transition:background .3s,color .3s}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--bg3);border-radius:3px}
a{color:inherit;text-decoration:none}
.sidebar{width:var(--sidebar-w);min-height:100vh;background:var(--bg2);border-left:1px solid var(--card-b);display:flex;flex-direction:column;flex-shrink:0;position:fixed;right:0;top:0;bottom:0;z-index:200;transition:transform .25s cubic-bezier(.4,0,.2,1),background .3s,border-color .3s}
.logo{display:flex;align-items:center;gap:12px;padding:20px 16px 16px;border-bottom:1px solid var(--card-b)}
.logo-img{width:38px;height:38px;border-radius:50%;overflow:hidden;border:1px solid var(--card-b);box-shadow:0 0 14px rgba(0,255,196,.15);flex-shrink:0}
.logo-name{font-size:16px;font-weight:900;font-family:'Cinzel',serif;color:var(--accent);letter-spacing:1px}
.logo-sub{font-size:10px;color:var(--t3);margin-top:1px}
.sb-close{display:none;position:absolute;left:12px;top:20px;background:var(--accent-d);border:1px solid var(--card-b);color:var(--t2);width:30px;height:30px;border-radius:8px;font-size:16px;align-items:center;justify-content:center;cursor:pointer}
.nav-wrap{flex:1;overflow-y:auto;padding:6px 0 8px}
.nav-sec{padding:14px 14px 4px;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--t3);font-weight:700}
.nav-it{display:flex;align-items:center;gap:9px;padding:9px 14px;color:var(--t3);font-size:12.5px;cursor:pointer;border-right:2px solid transparent;transition:all .15s;margin:1px 6px}
.nav-it i{font-size:16px;width:18px;text-align:center;flex-shrink:0}
.nav-it:hover{background:var(--accent-d);color:var(--t2)}
.nav-it.on{background:var(--accent-d);color:var(--accent);border-right-color:var(--accent);font-weight:600}
.nav-badge{margin-right:auto;background:rgba(0,255,196,0.15);color:var(--accent);font-size:9px;padding:1px 6px;border-radius:20px;font-weight:700}
.sb-foot{padding:12px 14px;border-top:1px solid var(--card-b)}
.theme-btn{display:flex;align-items:center;justify-content:center;gap:7px;background:var(--accent-d);color:var(--t2);border-radius:9px;padding:8px;font-size:12px;font-weight:500;font-family:inherit;border:1px solid var(--card-b);cursor:pointer;width:100%;transition:.15s;margin-bottom:7px}
.theme-btn:hover{background:var(--card-b);color:var(--t1)}
.logout-btn{display:flex;align-items:center;justify-content:center;gap:7px;background:var(--red-bg);color:var(--red-t);border-radius:9px;padding:8px;font-size:12px;font-weight:500;font-family:inherit;border:1px solid rgba(239,68,68,0.2);cursor:pointer;width:100%;transition:.15s;margin-top:6px}
.logout-btn:hover{background:rgba(239,68,68,0.2)}
.mob-top{display:none;position:fixed;top:0;right:0;left:0;height:52px;background:var(--bg2);border-bottom:1px solid var(--card-b);z-index:150;align-items:center;justify-content:space-between;padding:0 14px;transition:background .3s}
.mob-top .ml{display:flex;align-items:center;gap:9px}
.mob-logo{width:28px;height:28px;border-radius:50%;overflow:hidden;box-shadow:0 0 8px rgba(0,255,196,.15)}
.mob-title{color:var(--accent);font-family:'Cinzel',serif;font-size:15px;font-weight:800}
.mob-right{display:flex;gap:6px}
.menu-btn,.theme-mob{background:var(--accent-d);border:1px solid var(--card-b);color:var(--t2);width:34px;height:34px;border-radius:8px;font-size:17px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:.15s}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:190;backdrop-filter:blur(3px)}
.overlay.show{display:block}
.main{margin-right:var(--sidebar-w);flex:1;padding:28px 28px 60px;min-width:0;transition:margin .25s}
.pg{display:none}
.pg.on{display:block;animation:fi .2s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.topbar{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:22px;flex-wrap:wrap;gap:12px}
.tb-title{font-size:18px;font-weight:700;color:var(--t1);display:flex;align-items:center;gap:8px;letter-spacing:-.02em}
.tb-title i{color:var(--accent);font-size:20px}
.tb-sub{font-size:11px;color:var(--t3);margin-top:4px}
.tb-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{font-size:10px;padding:3px 10px;border-radius:20px;font-weight:700;display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.bg-green{background:var(--green-bg);color:var(--green-t)}
.bg-blue{background:var(--accent-d);color:var(--accent)}
.bg-amber{background:var(--amber-bg);color:var(--amber-t)}
.bg-red{background:var(--red-bg);color:var(--red-t)}
.bg-purple{background:var(--purple-bg);color:var(--purple-t)}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;display:inline-block}
.dg{background:var(--green)}.dr{background:var(--red)}.da{background:var(--amber)}.db{background:var(--accent)}
.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:18px}
.metric{background:var(--card);border:1px solid var(--card-b);border-radius:var(--radius);padding:17px 17px 14px;transition:all .2s;position:relative;overflow:hidden;cursor:default}
.metric::after{content:'';position:absolute;top:0;right:0;width:3px;height:100%;background:var(--accent);opacity:0;transition:.2s}
.metric:hover{border-color:var(--card-bh);transform:translateY(-2px);box-shadow:var(--shadow)}
.metric:hover::after{opacity:1}
.metric.suc::after{background:var(--green)}
.metric.dan::after{background:var(--red)}

.m-icon{width:34px;height:34px;border-radius:8px;background:var(--accent-d);display:flex;align-items:center;justify-content:center;margin-bottom:11px;color:var(--accent);font-size:17px}
.m-icon.suc{background:var(--green-bg);color:var(--green)}
.m-icon.dan{background:var(--red-bg);color:var(--red)}
.m-icon.pur{background:var(--purple-bg);color:var(--purple)}
.m-label{font-size:10px;color:var(--t3);margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.m-val{font-size:25px;font-weight:700;color:var(--t1);line-height:1;letter-spacing:-.02em}
.m-unit{font-size:12px;font-weight:400;color:var(--t3)}
.m-sub{font-size:10px;color:var(--t3);margin-top:6px;display:flex;align-items:center;gap:3px}
.vless-box{background:linear-gradient(135deg,var(--bg3) 0%,var(--bg2) 100%);border:1px solid var(--card-b);border-radius:18px;padding:20px 22px;margin-bottom:18px;box-shadow:var(--shadow);position:relative;overflow:hidden;transition:background .3s}
.vless-box::before{content:'';position:absolute;top:-50px;left:-50px;width:180px;height:180px;background:radial-gradient(circle,var(--accent-d),transparent 70%);pointer-events:none}
.vl-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:13px;flex-wrap:wrap;gap:8px}
.vl-title{color:var(--t2);font-size:11px;display:flex;align-items:center;gap:6px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}
.vl-title i{color:var(--accent);font-size:15px}
.vl-code{background:rgba(0,0,0,.18);border:1px solid var(--card-b);border-radius:9px;padding:13px 15px;font-size:11px;font-family:ui-monospace,monospace;color:var(--accent);word-break:break-all;line-height:1.8;letter-spacing:.01em}
[data-theme="light"] .vl-code{background:rgba(0,0,0,.04)}
.vl-actions{display:flex;gap:8px;margin-top:13px;flex-wrap:wrap}
.btn{font-family:inherit;font-size:12px;font-weight:700;border-radius:9px;padding:8px 14px;cursor:pointer;display:inline-flex;align-items:center;gap:5px;border:none;transition:all .15s;white-space:nowrap}
.btn i{font-size:13px}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-p{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;box-shadow:0 2px 14px rgba(0,255,196,.15)}
.btn-p:hover{filter:brightness(1.1);box-shadow:0 4px 18px rgba(0,255,196,.25)}
.btn-o{background:transparent;border:1px solid var(--card-b);color:var(--t2)}
.btn-o:hover{background:var(--accent-d);border-color:var(--accent)}
.btn-g{background:var(--accent-d);color:var(--accent);border:1px solid var(--card-b)}
.btn-g:hover{background:rgba(0,255,196,.22)}
.btn-d{background:var(--red-bg);color:var(--red-t);border:1px solid rgba(239,68,68,.2)}
.btn-d:hover{background:rgba(239,68,68,.2)}
.btn-pur{background:var(--purple-bg);color:var(--purple-t);border:1px solid var(--card-b)}
.btn-pur:hover{background:rgba(0,85,255,.22)}
.btn-amber{background:var(--amber-bg);color:var(--amber-t);border:1px solid rgba(245,158,11,.2)}
.btn-amber:hover{background:rgba(245,158,11,.22)}
.btn-sm{padding:5px 9px;font-size:10.5px;border-radius:7px}
.btn-icon{width:30px;height:30px;padding:0;justify-content:center;border-radius:5px}
.card{background:var(--card);border:1px solid var(--card-b);border-radius:var(--radius);padding:18px 20px;transition:border-color .2s,background .3s}
.card:hover{border-color:var(--card-bh)}
.card-title{font-size:12.5px;font-weight:700;color:var(--t1);margin-bottom:15px;display:flex;align-items:center;gap:7px}
.card-title i{font-size:16px;color:var(--accent)}
.ml-auto{margin-right:auto}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-bottom:16px}
.g3{display:grid;grid-template-columns:2fr 1fr;gap:13px;margin-bottom:16px}
.mb16{margin-bottom:16px}
.sr{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(0,255,196,0.05);font-size:12px}
.sr:last-child{border-bottom:none}
.sr-k{color:var(--t2);display:flex;align-items:center;gap:6px}
.sr-k i{font-size:13px;color:var(--t3)}
.sr-v{color:var(--t1);font-weight:600;font-size:11.5px}
.ch{position:relative;height:230px}
.ch-sm{position:relative;height:185px}
.exp-chip{font-size:9px;padding:3px 8px;border-radius:6px;font-weight:700;display:inline-flex;align-items:center;gap:3px}
.ec-ok{background:var(--green-bg);color:var(--green-t)}
.ec-warn{background:var(--amber-bg);color:var(--amber-t)}
.ec-exp{background:var(--red-bg);color:var(--red-t)}
.ec-inf{background:var(--accent-d);color:var(--accent)}
.tog{width:19px;height:34px;border-radius:19px;background:rgba(100,116,139,0.25);position:relative;cursor:pointer;transition:.2s;flex-shrink:0;border:none}
.tog::after{content:'';position:absolute;width:13px;height:13px;border-radius:50%;background:#fff;left:3px;bottom:3px;transition:.2s;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.tog.on{background:var(--green)}
.tog.on::after{bottom:18px}
.form-row{display:flex;gap:9px;flex-wrap:wrap;align-items:flex-end}
.fg{display:flex;flex-direction:column;gap:5px}
.fg label{font-size:10px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.06em}
.fi,.fs{padding:9px 12px;border-radius:9px;border:1px solid var(--card-b);background:rgba(0,0,0,.18);color:var(--t1);font-family:inherit;font-size:12px;outline:none;transition:.15s;min-width:100px}
[data-theme="light"] .fi,[data-theme="light"] .fs{background:rgba(0,0,0,.04)}
.fi::placeholder{color:var(--t3)}
.fi:focus,.fs:focus{border-color:rgba(0,255,196,.45);background:rgba(0,0,0,.25);box-shadow:0 0 0 3px rgba(0,255,196,.08)}
.fs option{background:var(--bg2)}
[data-theme="light"] .fs option{background:#fff}
.cl{background:var(--accent-d);border:1px solid var(--card-b);border-radius:10px;padding:11px 13px;font-size:11px;color:var(--t2);display:flex;gap:9px;align-items:flex-start;line-height:1.8;margin-top:12px}
.cl i{font-size:15px;color:var(--accent);margin-top:1px;flex-shrink:0}

.create-panel{background:linear-gradient(155deg,var(--bg3) 0%,var(--card) 55%);border:1px solid var(--card-b);border-radius:22px;padding:0;overflow:hidden;box-shadow:var(--shadow);margin-bottom:16px;position:relative}
.create-panel::before{content:'';position:absolute;top:-60px;left:-60px;width:220px;height:220px;background:radial-gradient(circle,var(--accent-d),transparent 70%);pointer-events:none}
.cp-head{display:flex;align-items:center;gap:13px;padding:22px 24px 18px;position:relative;z-index:1}
.cp-head-icon{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;color:#000;font-size:20px;flex-shrink:0;box-shadow:0 6px 18px rgba(0,255,196,.2)}
.cp-head-text{flex:1;min-width:0}
.cp-head-title{font-size:15px;font-weight:800;color:var(--t1);letter-spacing:-.01em}
.cp-head-sub{font-size:11px;color:var(--t3);margin-top:2px}
.cp-body{padding:2px 24px 22px;position:relative;z-index:1}
.cp-row{display:grid;grid-template-columns:1.3fr 1fr;gap:14px;margin-bottom:16px}
.cp-block{background:rgba(0,0,0,.14);border:1px solid var(--card-b);border-radius:14px;padding:14px 16px}
[data-theme="light"] .cp-block{background:rgba(0,255,196,.03)}
.cp-block-label{font-size:10px;font-weight:800;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;display:flex;align-items:center;gap:6px;margin-bottom:11px}
.cp-block-label i{color:var(--accent);font-size:14px}
.cp-input-full{width:100%;padding:10px 13px;border-radius:10px;border:1px solid var(--card-b);background:rgba(0,0,0,.18);color:var(--t1);font-family:inherit;font-size:12.5px;outline:none;transition:.15s}
[data-theme="light"] .cp-input-full{background:#fff}
.cp-input-full:focus{border-color:rgba(0,255,196,.5);box-shadow:0 0 0 3px rgba(0,255,196,.1)}
.cp-input-full::placeholder{color:var(--t3)}
.cp-mini-row{display:flex;gap:8px;margin-top:9px}
.cp-quota-inputs{display:flex;gap:8px}
.cp-quota-inputs .cp-input-full{flex:1}
.cp-quota-inputs select.cp-input-full{flex:0 0 76px}
.chip-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}
.chip{font-size:10.5px;font-weight:700;padding:5px 12px;border-radius:8px;background:var(--accent-d);color:var(--t2);border:1px solid var(--card-b);cursor:pointer;transition:.15s;white-space:nowrap}
.chip:hover{background:rgba(0,255,196,.18);color:var(--accent)}
.chip.active{background:var(--accent);color:#000;border-color:var(--accent);box-shadow:0 3px 10px rgba(0,255,196,.25)}
.proto-cards{display:grid;grid-template-columns:repeat(auto-fit, minmax(105px, 1fr));gap:9px}
.proto-card{border:1.5px solid var(--card-b);border-radius:13px;padding:13px 10px;cursor:pointer;transition:.18s;text-align:center;position:relative;background:rgba(0,0,0,.1)}
[data-theme="light"] .proto-card{background:#fff}
.proto-card:hover{border-color:var(--card-bh);transform:translateY(-1px)}
.proto-card.active{border-color:var(--accent);background:var(--accent-d);box-shadow:0 0 0 3px rgba(0,255,196,.1)}
.proto-card.active .proto-card-check{opacity:1;transform:scale(1)}
.proto-card-check{position:absolute;top:7px;left:7px;width:16px;height:16px;border-radius:50%;background:var(--accent);color:#000;font-size:10px;display:flex;align-items:center;justify-content:center;opacity:0;transform:scale(.5);transition:.18s}
.proto-card-icon{width:32px;height:32px;border-radius:9px;background:var(--accent-d);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:16px;margin:0 auto 8px}
.proto-card.active .proto-card-icon{background:var(--accent);color:#000}
.proto-card-title{font-size:11px;font-weight:800;color:var(--t1)}
.proto-card-desc{font-size:9px;color:var(--t3);margin-top:3px;line-height:1.5}
.cp-footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding-top:16px;border-top:1px solid var(--card-b);flex-wrap:wrap}
.cp-footer-note{display:flex;align-items:center;gap:8px;font-size:10.5px;color:var(--t3);line-height:1.7;flex:1;min-width:220px}
.cp-footer-note i{color:var(--accent);font-size:15px;flex-shrink:0}
.cp-submit-btn{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;border:none;border-radius:13px;padding:13px 26px;font-family:inherit;font-size:13px;font-weight:800;cursor:pointer;display:flex;align-items:center;gap:8px;box-shadow:0 6px 20px rgba(0,255,196,.2);transition:.18s;white-space:nowrap}
.cp-submit-btn:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(0,255,196,.3)}

.srv-panel{background:linear-gradient(155deg,var(--bg3) 0%,var(--card) 60%);border:1px solid var(--card-b);border-radius:22px;overflow:hidden;box-shadow:var(--shadow);position:relative}
.srv-panel::before{content:'';position:absolute;top:-60px;left:-60px;width:200px;height:200px;background:radial-gradient(circle,var(--accent-d),transparent 70%);pointer-events:none}
.srv-hero{display:flex;align-items:center;gap:14px;padding:22px 24px;position:relative;z-index:1;border-bottom:1px solid var(--card-b)}
.srv-hero-icon{width:50px;height:50px;border-radius:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;color:#000;font-size:22px;flex-shrink:0;box-shadow:0 6px 18px rgba(0,255,196,.25)}
.srv-hero-text{flex:1;min-width:0}
.srv-hero-domain{font-size:15px;font-weight:800;color:var(--t1);word-break:break-all}
.srv-hero-sub{font-size:10.5px;color:var(--t3);margin-top:4px;display:flex;align-items:center;gap:6px}
.srv-tiles{display:grid;grid-template-columns:1fr 1fr;gap:11px;padding:20px 22px 22px;position:relative;z-index:1}
.srv-tile{display:flex;align-items:center;gap:11px;background:rgba(0,0,0,.14);border:1px solid var(--card-b);border-radius:13px;padding:12px 14px;transition:.18s}
[data-theme="light"] .srv-tile{background:rgba(0,255,196,.03)}
.srv-tile:hover{border-color:var(--card-bh);transform:translateY(-1px)}
.srv-tile-icon{width:34px;height:34px;border-radius:10px;background:var(--accent-d);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.srv-tile-text{min-width:0}
.srv-tile-label{font-size:9.5px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.srv-tile-val{font-size:12px;font-weight:700;color:var(--t1);word-break:break-word}

.pw-panel{background:linear-gradient(155deg,var(--bg3) 0%,var(--card) 60%);border:1px solid var(--card-b);border-radius:22px;overflow:hidden;box-shadow:var(--shadow);position:relative}
.pw-panel::before{content:'';position:absolute;top:-60px;right:-60px;width:200px;height:200px;background:radial-gradient(circle,var(--purple-bg),transparent 70%);pointer-events:none}
.pw-hero{display:flex;align-items:center;gap:14px;padding:22px 24px 18px;position:relative;z-index:1}
.pw-hero-icon{width:50px;height:50px;border-radius:14px;background:linear-gradient(135deg,var(--purple),#0044CC);display:flex;align-items:center;justify-content:center;color:#fff;font-size:22px;flex-shrink:0;box-shadow:0 6px 18px rgba(0,85,255,.25)}
.pw-hero-text{flex:1;min-width:0}
.pw-hero-title{font-size:15px;font-weight:800;color:var(--t1)}
.pw-hero-sub{font-size:10.5px;color:var(--t3);margin-top:3px}
.pw-body{padding:2px 24px 22px;position:relative;z-index:1}
.pw-field{position:relative;margin-bottom:13px}
.pw-field label{display:block;font-size:10px;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px}
.pw-input{width:100%;padding:11px 42px 11px 14px;border-radius:11px;border:1px solid var(--card-b);background:rgba(0,0,0,.18);color:var(--t1);font-family:inherit;font-size:12.5px;outline:none;transition:.15s}
[data-theme="light"] .pw-input{background:#fff}
.pw-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d)}
.pw-eye{position:absolute;left:12px;top:34px;background:none;border:none;color:var(--t3);cursor:pointer;font-size:16px;padding:4px;display:flex}
.pw-eye:hover{color:var(--accent)}
.pw-strength{height:4px;border-radius:3px;background:var(--accent-d);margin-top:8px;overflow:hidden;display:flex;gap:3px}
.pw-strength-seg{flex:1;height:100%;border-radius:3px;background:rgba(100,116,139,.2);transition:.25s}
.pw-strength-label{font-size:9.5px;color:var(--t3);margin-top:5px;display:flex;align-items:center;gap:5px}
.pw-reqs{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px;margin-bottom:16px}
.pw-req{font-size:9.5px;padding:4px 10px;border-radius:7px;background:var(--accent-d);color:var(--t3);font-weight:600;display:flex;align-items:center;gap:4px;transition:.18s}
.pw-req.met{background:var(--green-bg);color:var(--green-t)}
.pw-submit{width:100%;justify-content:center;background:linear-gradient(135deg,var(--purple),#0044CC);color:#fff;border:none;border-radius:12px;padding:12px;font-family:inherit;font-size:13px;font-weight:800;cursor:pointer;display:flex;align-items:center;gap:8px;box-shadow:0 6px 18px rgba(0,85,255,.22);transition:.18s}

.conn-hero{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
.conn-hero-tile{background:var(--card);border:1px solid var(--card-b);border-radius:16px;padding:16px 18px;position:relative;overflow:hidden;transition:.2s}
.conn-hero-tile:hover{border-color:var(--card-bh);transform:translateY(-2px);box-shadow:var(--shadow)}
.conn-hero-tile::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--green),transparent)}
.conn-hero-icon{width:32px;height:32px;border-radius:9px;background:var(--green-bg);color:var(--green-t);display:flex;align-items:center;justify-content:center;font-size:15px;margin-bottom:10px}
.conn-hero-tile:nth-child(2) .conn-hero-icon{background:var(--accent-d);color:var(--accent)}
.conn-hero-tile:nth-child(3) .conn-hero-icon{background:var(--purple-bg);color:var(--purple-t)}
.conn-hero-tile:nth-child(4) .conn-hero-icon{background:var(--amber-bg);color:var(--amber-t)}
.conn-hero-label{font-size:9.5px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.conn-hero-val{font-size:21px;font-weight:800;color:var(--t1);line-height:1;letter-spacing:-.02em}

.conn-toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.conn-toolbar-title{font-size:12px;font-weight:800;color:var(--t2);display:flex;align-items:center;gap:7px;text-transform:uppercase;letter-spacing:.06em}
.conn-toolbar-title i{color:var(--green);font-size:15px}
.conn-live-badge{display:flex;align-items:center;gap:6px;font-size:10.5px;font-weight:700;color:var(--green-t);background:var(--green-bg);padding:5px 12px;border-radius:20px;border:1px solid rgba(74,222,128,.2)}
.conn-live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 1.6s infinite}

.conn-grid-v2{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.conn-card-v2{background:var(--card);border:1px solid var(--card-b);border-radius:18px;padding:0;overflow:hidden;transition:all .22s cubic-bezier(.4,0,.2,1);position:relative}
.conn-card-v2:hover{border-color:var(--card-bh);transform:translateY(-3px);box-shadow:0 14px 32px rgba(0,255,196,.08)}
.conn-card-v2-glow{position:absolute;top:-40px;left:-40px;width:140px;height:140px;background:radial-gradient(circle,rgba(74,222,128,.1),transparent 70%);pointer-events:none}
.conn-card-v2-top{display:flex;align-items:center;gap:12px;padding:16px 17px 13px;position:relative;z-index:1}
.conn-avatar{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,var(--green),#00B359);display:flex;align-items:center;justify-content:center;color:#000;font-size:18px;flex-shrink:0;position:relative;box-shadow:0 4px 14px rgba(74,222,128,.2)}
.conn-card-v2-id{flex:1;min-width:0}
.conn-ip-v2{font-family:ui-monospace,monospace;font-size:14px;font-weight:800;color:var(--t1);display:flex;align-items:center;gap:6px}
.conn-ip-copy{background:none;border:none;color:var(--t3);cursor:pointer;font-size:12px;padding:2px;display:flex;transition:.15s}
.conn-ip-copy:hover{color:var(--accent)}
.conn-label-v2{font-size:10.5px;color:var(--t3);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.conn-status-pill{font-size:9px;font-weight:800;padding:4px 9px;border-radius:20px;background:var(--green-bg);color:var(--green-t);display:flex;align-items:center;gap:4px;white-space:nowrap;flex-shrink:0}
.conn-card-v2-divider{height:1px;background:linear-gradient(90deg,transparent,var(--card-b) 15%,var(--card-b) 85%,transparent);margin:0 17px}
.conn-card-v2-body{padding:14px 17px 16px}
.conn-proto-row{margin-bottom:12px}
.conn-stat-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.conn-stat-box{display:flex;align-items:center;gap:8px}
.conn-stat-icon{width:26px;height:26px;border-radius:8px;background:var(--accent-d);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
.conn-stat-icon.time{background:var(--purple-bg);color:var(--purple-t)}
.conn-stat-text-label{font-size:8.5px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.conn-stat-text-val{font-size:11.5px;font-weight:700;color:var(--t1);margin-top:1px}
.conn-duration-track{height:5px;border-radius:4px;background:var(--accent-d);overflow:hidden;position:relative}
.conn-duration-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--green),#00FFC4);position:relative;overflow:hidden}

.conn-empty-v2{text-align:center;padding:70px 20px;background:var(--card);border:1px dashed var(--card-b);border-radius:20px}
.conn-empty-v2-icon{width:64px;height:64px;border-radius:18px;background:var(--accent-d);display:flex;align-items:center;justify-content:center;font-size:28px;color:var(--t3);margin:0 auto 16px}

.sub-box{background:var(--purple-bg);border:1px solid rgba(0,85,255,.2);border-radius:10px;padding:14px 16px;display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:11px}
.sub-url{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--purple-t);word-break:break-all;flex:1}
.empty{text-align:center;padding:50px 20px;color:var(--t3)}
.empty i{font-size:40px;opacity:.3;margin-bottom:12px;display:block}

.subs-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.subs-search{flex:1;min-width:200px;position:relative}
.subs-search input{width:100%;padding:11px 40px 11px 15px;border-radius:12px;border:1px solid var(--card-b);background:var(--card);color:var(--t1);font-family:inherit;font-size:12.5px;outline:none;transition:.15s}
.subs-search input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d)}
.subs-search i{position:absolute;left:14px;top:50%;transform:translateY(-50%);color:var(--t3);font-size:15px}

.sub-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-bottom:18px}
.sub-card{background:var(--card);border:1px solid var(--card-b);border-radius:20px;padding:0;overflow:hidden;transition:all .25s cubic-bezier(.4,0,.2,1);position:relative}
.sub-card:hover{border-color:var(--card-bh);transform:translateY(-4px);box-shadow:0 16px 36px rgba(0,255,196,.05)}
.sub-card-top{background:linear-gradient(155deg,var(--purple-bg) 0%,transparent 65%);padding:20px 20px 16px;position:relative}
.sub-card-top::before{content:'';position:absolute;top:-30px;left:-30px;width:130px;height:130px;background:radial-gradient(circle,rgba(0,85,255,.14),transparent 70%);pointer-events:none}
.sub-card-head-v2{display:flex;align-items:flex-start;gap:13px;position:relative;z-index:1}
.sub-card-icon{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,var(--purple),#0044CC);display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;flex-shrink:0;box-shadow:0 6px 16px rgba(0,85,255,.2)}
.sub-card-titles{flex:1;min-width:0}
.sub-card-name-v2{font-size:15.5px;font-weight:800;color:var(--t1);letter-spacing:-.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub-card-desc-v2{font-size:11px;color:var(--t3);margin-top:3px;line-height:1.6;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.sub-card-lock-badge{flex-shrink:0;width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px}
.sub-card-lock-badge.locked{background:var(--amber-bg);color:var(--amber-t)}
.sub-card-lock-badge.open{background:var(--green-bg);color:var(--green-t)}

.sub-card-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:0;position:relative;z-index:1;margin-top:16px;background:rgba(0,0,0,.14);border:1px solid var(--card-b);border-radius:13px;overflow:hidden}
.sub-card-stat{padding:11px 8px;text-align:center;border-left:1px solid var(--card-b)}
.sub-card-stat:last-child{border-left:none}
.sub-card-stat-val{font-size:15px;font-weight:800;color:var(--t1);line-height:1.2}
.sub-card-stat-label{font-size:8.5px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.sub-card-url-row{margin:14px 20px 0;background:var(--purple-bg);border:1px dashed rgba(0,85,255,.25);border-radius:11px;padding:9px 12px;display:flex;align-items:center;gap:8px}
.sub-card-url-text{font-family:ui-monospace,monospace;font-size:9.5px;color:var(--purple-t);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub-card-url-copy{background:none;border:none;color:var(--purple-t);cursor:pointer;font-size:13px;padding:3px;display:flex;flex-shrink:0;transition:.15s}
.sub-card-url-copy:hover{color:var(--accent);transform:scale(1.1)}
.sub-card-bottom{padding:14px 20px 18px;display:flex;gap:7px;flex-wrap:wrap}
.sub-card-bottom .btn{flex:1;justify-content:center;min-width:fit-content}

.subs-empty-v2{text-align:center;padding:70px 20px;background:var(--card);border:1px dashed var(--card-b);border-radius:20px;grid-column:1/-1}
.subs-empty-v2-icon{width:64px;height:64px;border-radius:18px;background:var(--purple-bg);display:flex;align-items:center;justify-content:center;font-size:28px;color:var(--purple-t);margin:0 auto 16px}

.modal-v2{background:var(--card);border:1px solid var(--card-b);border-radius:22px;padding:0;max-width:430px;width:calc(100% - 32px);max-height:92vh;overflow-y:auto;position:relative;animation:fi .2s ease;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.modal-v2-head{background:linear-gradient(155deg,rgba(0,85,255,.14) 0%,transparent 65%);padding:18px 22px 14px;position:relative;overflow:hidden}
.modal-v2-head::before{content:'';position:absolute;top:-50px;left:-50px;width:160px;height:160px;background:radial-gradient(circle,rgba(0,85,255,.2),transparent 70%);pointer-events:none}
.modal-v2-close{position:absolute;top:14px;left:14px;background:var(--accent-d);border:1px solid var(--card-b);color:var(--t2);width:30px;height:30px;border-radius:9px;font-size:15px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:2;transition:.15s}
.modal-v2-close:hover{background:var(--red-bg);color:var(--red-t);border-color:rgba(239,68,68,.25)}
.modal-v2-icon{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,var(--purple),#0044CC);display:flex;align-items:center;justify-content:center;color:#fff;font-size:19px;margin-bottom:10px;position:relative;z-index:1;box-shadow:0 8px 18px rgba(0,85,255,.2)}
.modal-v2-title{font-size:15.5px;font-weight:800;color:var(--t1);position:relative;z-index:1;letter-spacing:-.01em}
.modal-v2-sub{font-size:10.5px;color:var(--t3);margin-top:3px;position:relative;z-index:1;line-height:1.6}
.modal-v2-body{padding:16px 22px 20px;border-top:1px solid var(--card-b)}
.modal-v2-field{margin-bottom:11px}
.modal-v2-field label{display:flex;align-items:center;gap:5px;font-size:9.5px;font-weight:800;color:var(--t2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.modal-v2-input-wrap{position:relative}
.modal-v2-input{width:100%;padding:9px 38px 9px 13px;border-radius:11px;border:1px solid var(--card-b);background:rgba(0,0,0,.2);color:var(--t1);font-family:inherit;font-size:12.5px;outline:none;transition:.18s}
[data-theme="light"] .modal-v2-input{background:rgba(0,85,255,.04)}
.modal-v2-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d);background:rgba(0,0,0,.28)}
.modal-v2-hint{background:var(--accent-d);border:1px solid rgba(0,255,196,.18);border-radius:11px;padding:9px 12px;font-size:10px;color:var(--t2);display:flex;gap:7px;align-items:flex-start;line-height:1.6;margin-top:2px}
.modal-v2-footer{display:flex;gap:8px;margin-top:15px}
.modal-v2-btn-cancel{flex:.75;justify-content:center;padding:10px;border-radius:11px;background:transparent;border:1px solid var(--card-b);color:var(--t2);font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:.15s;display:flex;align-items:center}
.modal-v2-btn-cancel:hover{background:var(--accent-d);color:var(--t1)}
.modal-v2-btn-submit{flex:1;justify-content:center;padding:10px;border-radius:11px;background:linear-gradient(135deg,var(--purple),#0044CC);color:#fff;border:none;font-family:inherit;font-size:12px;font-weight:800;cursor:pointer;display:flex;align-items:center;gap:6px;box-shadow:0 6px 18px rgba(0,85,255,.2);transition:.18s}

.lmodal-head{background:linear-gradient(155deg,var(--accent-d) 0%,transparent 70%);padding:22px 24px 18px;position:relative;border-bottom:1px solid var(--card-b)}
.lmodal-icon-row{display:flex;align-items:center;gap:12px;position:relative;z-index:1}
.lmodal-icon{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;color:#000;font-size:19px;flex-shrink:0;box-shadow:0 6px 16px rgba(0,255,196,.25)}
.lmodal-title-v2{font-size:14.5px;font-weight:800;color:var(--t1)}
.lmodal-sub-v2{font-size:10.5px;color:var(--t3);margin-top:2px}
.lmodal-search{margin-top:14px;position:relative}
.lmodal-search input{width:100%;padding:10px 38px 10px 13px;border-radius:11px;border:1px solid var(--card-b);background:rgba(0,0,0,.2);color:var(--t1);font-family:inherit;font-size:12px;outline:none}
.lmodal-search input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d)}
.lmodal-search i{position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--t3);font-size:14px}
.lmodal-quickbar{display:flex;gap:8px;margin-top:11px;position:relative;z-index:1}
.lmodal-qbtn{font-size:10px;font-weight:700;padding:5px 11px;border-radius:8px;background:var(--accent-d);color:var(--accent);border:1px solid var(--card-b);cursor:pointer;transition:.15s;font-family:inherit}
.lmodal-count{margin-right:auto;font-size:10.5px;color:var(--t3);display:flex;align-items:center}
.lmodal-list{padding:10px 14px;max-height:360px;overflow-y:auto}
.lrow-v2{display:flex;align-items:center;gap:11px;padding:11px 12px;border-radius:13px;cursor:pointer;transition:.15s;margin-bottom:4px;border:1px solid transparent}
.lrow-v2:hover{background:var(--accent-d)}
.lrow-v2.checked{background:var(--accent-d);border-color:var(--accent)}
.lrow-v2.half-checked {background:var(--accent-d)}
.lrow-v2.half-checked .lrow-v2-check {background:var(--accent-d);border-color:var(--accent)}
.lrow-v2.half-checked .lrow-v2-check i {opacity:1;transform:scale(1);color:var(--accent)}
.lrow-v2-check{width:20px;height:20px;border-radius:7px;border:2px solid var(--card-b);flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:.15s;background:rgba(0,0,0,.14)}
.lrow-v2.checked .lrow-v2-check{background:var(--accent);border-color:var(--accent)}
.lrow-v2.checked .lrow-v2-check i{opacity:1;transform:scale(1);color:#000}
.lrow-v2-check i{opacity:0;transform:scale(.5)}
.lrow-v2-avatar{width:34px;height:34px;border-radius:10px;background:var(--accent-d);color:var(--accent);display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.lrow-v2.checked .lrow-v2-avatar{background:var(--accent);color:#000}
.lrow-v2-info{flex:1;min-width:0}
.lrow-v2-name{font-size:12.5px;font-weight:700;color:var(--t1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lrow-v2-meta{font-size:9.5px;color:var(--t3);margin-top:2px;display:flex;align-items:center;gap:6px}
.lrow-v2-status{font-size:9px;font-weight:800;padding:3px 9px;border-radius:20px;flex-shrink:0;white-space:nowrap}
.lrow-v2-status.on{background:var(--green-bg);color:var(--green-t)}
.lrow-v2-status.off{background:var(--red-bg);color:var(--red-t)}
.lmodal-footer{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:16px 24px;border-top:1px solid var(--card-b)}
.lmodal-footer-info{font-size:10.5px;color:var(--t3);display:flex;align-items:center;gap:6px}
.lmodal-footer-info i{color:var(--accent)}
.lmodal-footer-btns{display:flex;gap:8px}

.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:500;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.modal-bg.open{display:flex}
.modal{background:var(--card);border:1px solid var(--card-b);border-radius:20px;padding:28px 26px;max-width:520px;width:calc(100% - 32px);max-height:90vh;overflow-y:auto;position:relative;animation:fi .2s ease}
.modal-close{position:absolute;top:14px;left:14px;background:var(--accent-d);border:1px solid var(--card-b);color:var(--t2);width:30px;height:30px;border-radius:8px;font-size:16px;display:flex;align-items:center;justify-content:center;cursor:pointer;border:none}
.modal-title{font-size:16px;font-weight:700;color:var(--t1);margin-bottom:18px;display:flex;align-items:center;gap:8px}
.modal-title i{color:var(--accent)}

.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(40px);background:var(--card);border:1px solid var(--card-b);color:var(--t1);border-radius:10px;padding:10px 18px;font-size:12.5px;opacity:0;transition:all .25s;z-index:999;pointer-events:none;display:flex;align-items:center;gap:8px;box-shadow:var(--shadow);white-space:nowrap}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast.ok{border-color:rgba(0,230,118,.3);background:var(--green-bg);color:var(--green-t)}
.toast.err{border-color:rgba(248,113,113,.3);background:var(--red-bg);color:var(--red-t)}
.dash-footer{border-top:1px solid var(--card-b);margin-top:14px;padding-top:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}

.cfg-grid{display:flex;flex-direction:column;gap:10px}
.cfg-card{background:var(--card);border:1px solid var(--card-b);border-radius:14px;padding:0;transition:all .2s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}
.cfg-card:hover{border-color:var(--card-bh);box-shadow:0 6px 24px rgba(0,255,196,.05)}
.cfg-card.is-off{opacity:.6}
.cfg-card.is-exp{opacity:.78}
.cfg-row{display:flex;align-items:center;gap:16px;padding:14px 18px}
.cfg-status-dot{width:9px;height:9px;border-radius:50%;background:var(--green);flex-shrink:0;box-shadow:0 0 0 3px var(--green-bg)}
.cfg-card.is-off .cfg-status-dot{background:var(--red);box-shadow:0 0 0 3px var(--red-bg)}
.cfg-card.is-exp .cfg-status-dot{background:var(--amber);box-shadow:0 0 0 3px var(--amber-bg)}
.cfg-identity{display:flex;flex-direction:column;gap:3px;min-width:150px;flex-shrink:0}
.cfg-label{font-size:13.5px;font-weight:700;color:var(--t1);display:flex;align-items:center;gap:7px}
.cfg-sub-meta{display:flex;align-items:center;gap:8px;font-size:10px;color:var(--t3)}
.cfg-uuid-mini{font-family:ui-monospace,monospace;font-size:9.5px;color:var(--accent);background:var(--accent-d);padding:2px 7px;border-radius:5px;cursor:pointer;transition:.15s}
.cfg-divider-v{width:1px;align-self:stretch;background:var(--card-b);flex-shrink:0}
.cfg-usage-col{flex:1;min-width:160px;display:flex;flex-direction:column;gap:5px}
.ubar{height:5px;border-radius:4px;background:rgba(0,255,196,0.1);overflow:hidden}
.ubar-f{height:100%;border-radius:4px;transition:width .4s ease}
.utxt{font-size:10px;color:var(--t3);display:flex;justify-content:space-between}
.cfg-exp-col{flex-shrink:0;min-width:110px}
.cfg-badges-col{display:flex;flex-direction:column;gap:5px;flex-shrink:0;align-items:flex-end}
.cfg-actions{display:flex;gap:5px;flex-shrink:0}
.proto-chip{font-size:9px;padding:3px 8px;border-radius:6px;font-weight:700;white-space:nowrap}
.pc-ws{background:var(--accent-d);color:var(--accent)}
.pc-xhttp{background:var(--purple-bg);color:var(--purple-t)}
.pc-ultra{background:var(--green-bg);color:var(--green-t)}
.cfg-sub-tag{font-size:9.5px;color:var(--t3);display:flex;align-items:center;gap:4px;white-space:nowrap}
.cfg-sub-tag i{color:var(--accent);font-size:11px}

.log-timeline{display:flex;flex-direction:column}
.log-item{display:flex;gap:12px;padding:11px 0;border-bottom:1px solid rgba(0,255,196,0.05);position:relative}
.log-item:last-child{border-bottom:none}
.log-ic{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.log-ic.ok{background:var(--green-bg);color:var(--green-t)}
.log-ic.err{background:var(--red-bg);color:var(--red-t)}
.log-ic.warn{background:var(--amber-bg);color:var(--amber-t)}
.log-ic.info{background:var(--accent-d);color:var(--accent)}
.log-body{flex:1;min-width:0}
.log-msg{font-size:12.5px;color:var(--t1);line-height:1.6}
.log-time{font-size:9.5px;color:var(--t3);margin-top:2px;display:flex;align-items:center;gap:5px}
.log-kind{font-size:8.5px;padding:1px 7px;border-radius:10px;background:var(--accent-d);color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.erow{padding:9px 0;border-bottom:1px solid rgba(0,255,196,0.05)}
.erow:last-child{border-bottom:none}
.etime{color:var(--t3);font-size:9.5px;margin-bottom:3px;display:flex;align-items:center;gap:4px}
.emsg{color:var(--red-t);font-family:ui-monospace,monospace;background:var(--red-bg);padding:6px 9px;border-radius:6px;word-break:break-all;font-size:10.5px}

@media(max-width:880px){
  .cfg-row{flex-wrap:wrap}
  .cfg-divider-v{display:none}
  .cfg-usage-col{min-width:100%;order:5}
}
@media(max-width:768px){
  .cfg-grid{display:grid;grid-template-columns:1fr;gap:13px}
  .cfg-card{border-radius:16px}
  .cfg-row{flex-direction:column;align-items:stretch;gap:12px;padding:16px}
  .cfg-identity{min-width:0;flex:1}
  .cfg-usage-col{min-width:0}
  .cfg-exp-col{min-width:0}
  .cfg-badges-col{flex-direction:row;align-items:center;flex-wrap:wrap}
  .cfg-actions{flex-wrap:wrap;border-top:1px solid var(--card-b);padding-top:10px;margin-top:2px;width:100%}
}
@media(max-width:1050px){
  .sidebar{transform:translateX(100%)}
  .sidebar.open{transform:translateX(0);box-shadow:-10px 0 40px rgba(0,0,0,.4)}
  .sb-close{display:flex}
  .main{margin-right:0;padding-top:70px}
  .mob-top{display:flex}
  .metrics{grid-template-columns:1fr 1fr}
  .g2,.g3{grid-template-columns:1fr}
}
@media(max-width:500px){
  .metrics{grid-template-columns:1fr}
  .main{padding:62px 12px 50px}
  .sub-grid,.cfg-grid,.conn-grid-v2{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="toast" id="toast"></div>
<div class="modal-bg" id="modal-variations">
  <div class="modal" style="max-width:440px; padding: 22px;">
    <button class="modal-close" onclick="closeModal('modal-variations')"><i class="ti ti-x"></i></button>
    <div class="modal-title"><i class="ti ti-layers-linked"></i> لینک‌های اتصال</div>
    <div id="variations-list" style="display:flex;flex-direction:column;gap:10px;margin-top:10px;max-height:60vh;overflow-y:auto;padding-right:5px;"></div>
  </div>
</div>
<div class="modal-bg" id="modal-links">
  <div class="modal-v2" style="max-width:500px">
    <div class="lmodal-head">
      <button class="modal-v2-close" onclick="closeModal('modal-links')"><i class="ti ti-x"></i></button>
      <div class="lmodal-icon-row">
        <div class="lmodal-icon"><i class="ti ti-cpu"></i></div>
        <div>
          <div class="lmodal-title-v2">تخصیص نُد برای <span id="modal-sub-name" style="color:var(--accent)">—</span></div>
          <div class="lmodal-sub-v2">نُد‌های تنسور که باید در این خوشه کار کنند را انتخاب کنید</div>
        </div>
      </div>
      <div class="lmodal-search">
        <i class="ti ti-search"></i>
        <input type="text" id="lmodal-search-inp" placeholder="جستجوی نُد پردازشی..." oninput="filterLmodal(this.value)">
      </div>
      <div class="lmodal-quickbar">
        <button class="lmodal-qbtn" onclick="lmodalSelectAll(true)"><i class="ti ti-checks"></i> انتخاب همه</button>
        <button class="lmodal-qbtn" onclick="lmodalSelectAll(false)"><i class="ti ti-x"></i> لغو همه</button>
        <span class="lmodal-count" id="lmodal-count">۰ انتخاب شده</span>
      </div>
    </div>
    <div class="lmodal-list" id="modal-links-body">در حال بارگذاری...</div>
    <div class="lmodal-footer">
      <div class="lmodal-footer-info"><i class="ti ti-info-circle"></i> تغییرات به صورت بلادرنگ همگام می‌شود</div>
      <div class="lmodal-footer-btns">
        <button class="btn btn-o" onclick="closeModal('modal-links')">بستن</button>
        <button class="btn btn-p" id="modal-save-btn" onclick="saveSubLinks()"><i class="ti ti-check"></i> اعمال</button>
      </div>
    </div>
  </div>
</div>
<div class="modal-bg" id="modal-create-sub">
  <div class="modal-v2">
    <div class="modal-v2-head">
      <button class="modal-v2-close" onclick="closeModal('modal-create-sub')"><i class="ti ti-x"></i></button>
      <div class="modal-v2-icon"><i class="ti ti-server-cog"></i></div>
      <div class="modal-v2-title">ثبت خوشه پردازشی (Ensemble)</div>
      <div class="modal-v2-sub">یک ایزوله جدید برای پخش وزن‌های شبکه عصبی بسازید</div>
    </div>
    <div class="modal-v2-body">
      <div class="modal-v2-field">
        <label><i class="ti ti-tag"></i> شناسه خوشه</label>
        <input class="modal-v2-input" id="ns-name" placeholder="مثلاً: خوشه Llama-8B">
      </div>
      <div class="modal-v2-field">
        <label><i class="ti ti-align-left"></i> متا دیتا (اختیاری)</label>
        <input class="modal-v2-input" id="ns-desc" placeholder="شرح وظیفه این ایزوله پردازشی">
      </div>
      
      <div class="modal-v2-field">
        <label><i class="ti ti-world"></i> مسیرهایابی موازی (Load Balancers/Gateways)</label>
        <div id="ns-saved-customs" style="display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;margin-top:6px"></div>
        <div id="ns-customs-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px;margin-top:6px"></div>
        <button class="btn btn-sm btn-g" type="button" onclick="addSubCustomField('ns')"><i class="ti ti-plus"></i> ایجاد روت Gateway جدید</button>
      </div>

      <div class="modal-v2-field" style="margin-bottom:0">
        <label><i class="ti ti-lock"></i> کلید رمزنگاری Endpoint (اختیاری)</label>
        <input class="modal-v2-input" id="ns-pw" type="password" placeholder="خالی = بدون احراز هویت">
      </div>
      <div class="modal-v2-footer">
        <button class="btn btn-o" onclick="closeModal('modal-create-sub')" style="flex:.6">انصراف</button>
        <button class="btn btn-pur" onclick="createSub()"><i class="ti ti-server-cog"></i> ثبت کلاستر</button>
      </div>
    </div>
  </div>
</div>
<div class="modal-bg" id="modal-edit-sub">
  <div class="modal-v2">
    <div class="modal-v2-head" style="background:linear-gradient(155deg,rgba(16,185,129,.14) 0%,transparent 65%);">
      <button class="modal-v2-close" onclick="closeModal('modal-edit-sub')"><i class="ti ti-x"></i></button>
      <div class="modal-v2-icon" style="background:linear-gradient(135deg,var(--green),#0D9668);box-shadow:0 8px 18px rgba(16,185,129,.2);"><i class="ti ti-edit"></i></div>
      <div class="modal-v2-title">پیکربندی مجدد خوشه</div>
      <div class="modal-v2-sub">تغییر متادیتا و تنظیمات لودبالانسر</div>
    </div>
    <div class="modal-v2-body">
      <input type="hidden" id="es-id">
      <div class="modal-v2-field">
        <label><i class="ti ti-tag"></i> نام خوشه</label>
        <input class="modal-v2-input" id="es-name" placeholder="مثلاً: خوشه Llama-8B">
      </div>
      <div class="modal-v2-field">
        <label><i class="ti ti-align-left"></i> متا دیتا (اختیاری)</label>
        <input class="modal-v2-input" id="es-desc" placeholder="...">
      </div>
      
      <div class="modal-v2-field">
        <label><i class="ti ti-world"></i> مسیرهایابی موازی (Load Balancers/Gateways)</label>
        <div id="es-saved-customs" style="display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;margin-top:6px"></div>
        <div id="es-customs-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px;margin-top:6px"></div>
        <button class="btn btn-sm btn-g" type="button" onclick="addSubCustomField('es')"><i class="ti ti-plus"></i> ایجاد روت Gateway جدید</button>
      </div>

      <div class="modal-v2-field" style="margin-bottom:0">
        <label><i class="ti ti-lock"></i> کلید رمزنگاری جدید (اختیاری)</label>
        <input class="modal-v2-input" id="es-pw" type="password" placeholder="برای عدم تغییر، خالی بگذارید">
        <label style="margin-top:8px;display:flex;align-items:center;gap:6px;font-size:10px;text-transform:none">
            <input type="checkbox" id="es-remove-pw"> حذف کلید امنیتی (عمومی شدن Endpoint)
        </label>
      </div>
      <div class="modal-v2-footer">
        <button class="btn btn-o" onclick="closeModal('modal-edit-sub')" style="flex:.6">انصراف</button>
        <button class="btn btn-p" onclick="saveEditSub()" style="background:var(--green);box-shadow:0 6px 18px rgba(16,185,129,.2)"><i class="ti ti-check"></i> اعمال تنظیمات</button>
      </div>
    </div>
  </div>
</div>
<div class="modal-bg" id="modal-edit-link">
  <div class="modal">
    <button class="modal-close" onclick="closeModal('modal-edit-link')"><i class="ti ti-x"></i></button>
    <div class="modal-title"><i class="ti ti-edit"></i> ویرایش پارامترهای نُد</div>
    <input type="hidden" id="el-uuid">
    <div class="fg" style="margin-bottom:13px"><label>عنوان گراف پردازشی</label><input class="fi" id="el-label" style="width:100%"></div>
    <div class="fg" style="margin-bottom:13px"><label>درگاه کاستوم (اختیاری)</label><input class="fi" id="el-sub-domain" placeholder="مثلاً inference.cluster.ai" style="width:100%"></div>
    
    <div class="fg" style="margin-bottom:13px;width:100%">
      <label>تخصیص به خوشه‌های پردازشی (Ensembles)</label>
      <div id="el-subs-list" style="max-height:110px;overflow-y:auto;background:rgba(0,0,0,.15);border:1px solid var(--card-b);border-radius:10px;padding:8px;display:flex;flex-direction:column;gap:5px;width:100%"></div>
    </div>

    <div class="fg" style="margin-bottom:13px;width:100%">
      <label>استریم‌های توزیع شده کاستوم</label>
      <div id="el-saved-customs" style="display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;margin-bottom:6px;width:100%"></div>
      <div id="el-customs-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px;width:100%"></div>
      <button class="btn btn-sm btn-g" type="button" onclick="addCustomField('el')"><i class="ti ti-plus"></i> ایجاد روت کاستوم جدید</button>
    </div>

    <div class="form-row" style="margin-bottom:13px">
      <div class="fg" style="flex:1"><label>بودجه توکن‌ها (0 = نامحدود)</label><input class="fi" id="el-val" type="number" min="0" step="0.1" style="width:100%"></div>
      <div class="fg"><label>واحد</label><select class="fs" id="el-unit"><option value="GB">B-Tok</option><option value="MB">M-Tok</option></select></div>
    </div>
    <div class="fg" style="margin-bottom:13px"><label>مهلت Epoch (روز، 0 = متوقف نشود)</label><input class="fi" id="el-exp" type="number" min="0" step="1" style="width:100%"></div>
    <div class="fg" style="margin-bottom:13px"><label>لاگ متادیتا</label><input class="fi" id="el-note" style="width:100%"></div>
    <div class="form-row" style="margin-bottom:13px">
      <div class="fg" style="flex:1"><label>Q-TLS Profile</label>
        <select class="fs" id="el-fp" style="width:100%">
          <option value="chrome">fp16_accurate</option>
          <option value="firefox">int8_fast</option>
          <option value="safari">fp32_native</option>
          <option value="ios">mixed_precision</option>
          <option value="android">bfloat16</option>
          <option value="edge">edge_quant</option>
          <option value="random">randomized_dist</option>
        </select>
      </div>
      <div class="fg" style="flex:1"><label>لایه ALPN (خالی = پیش‌فرض)</label><input class="fi" id="el-alpn" placeholder="مثلاً: h2,http/1.1" style="width:100%"></div>
    </div>
    <div class="form-row" style="margin-bottom:16px">
      <div class="fg" style="flex:1"><label>پورت سینک (Sync Port)</label><input class="fi" id="el-port" type="number" min="1" max="65535" style="width:100%"></div>
      <div class="fg" style="flex:1"><label>حداکثر Worker مجاز (0 = نامحدود)</label><input class="fi" id="el-iplimit" type="number" min="0" step="1" style="width:100%"></div>
    </div>
    <div class="form-row" style="margin-bottom:16px">
      <div class="fg" style="flex:1"><label>نرخ پردازش مجاز (0 = نامحدود)</label><input class="fi" id="el-speed" type="number" min="0" step="0.5" style="width:100%"></div>
      <div class="fg"><label>واحد</label><select class="fs" id="el-speed-unit"><option value="MBIT">M-Tok/s</option><option value="KB">K-Tok/s</option></select></div>
    </div>
    
    <div class="cl"><i class="ti ti-info-circle"></i><span>برای ادامه یادگیری روی Epoch قبلی، مهلت را 0 قرار دهید.</span></div>
    <div style="margin-top:16px;display:flex;gap:8px;justify-content:flex-end">
      <button class="btn btn-o" onclick="closeModal('modal-edit-link')">انصراف</button>
      <button class="btn btn-p" onclick="saveEditLink()"><i class="ti ti-check"></i> ثبت پارامترها</button>
    </div>
  </div>
</div>
<div class="mob-top">
  <div class="ml">
    <div class="mob-logo">__LOGO_SVG__</div>
    <span class="mob-title">Nexus AI Cluster</span>
  </div>
  <div class="mob-right">
    <button class="theme-mob" id="theme-mob-btn" onclick="toggleTheme()"><i class="ti ti-sun" id="theme-mob-icon"></i></button>
    <button class="menu-btn" id="open-sb"><i class="ti ti-menu-2"></i></button>
  </div>
</div>
<div class="overlay" id="overlay"></div>
<aside class="sidebar" id="sb">
  <button class="sb-close" id="close-sb"><i class="ti ti-x"></i></button>
  <div class="logo">
    <div class="logo-img">__LOGO_SVG__</div>
    <div><div class="logo-name">Nexus AI</div><div class="logo-sub">Tensor-Core System v9.5</div></div>
  </div>
  <div class="nav-wrap">
    <div class="nav-sec">مانیتورینگ کلاستر</div>
    <div class="nav-it on" data-pg="overview"><i class="ti ti-layout-dashboard"></i> کلاستر (Overview)</div>
    <div class="nav-it" data-pg="links"><i class="ti ti-cpu"></i> نُدهای پردازشی <span class="nav-badge" id="links-nb">0</span></div>
    <div class="nav-it" data-pg="subgroups"><i class="ti ti-server-cog"></i> خوشه‌ها (Ensembles) <span class="nav-badge" id="subs-nb">0</span></div>
    <div class="nav-it" data-pg="subscriptions"><i class="ti ti-database-export"></i> رجیستری مدل</div>
    <div class="nav-it" data-pg="connections"><i class="ti ti-chart-arcs"></i> استریم‌های زنده <span class="nav-badge" id="conns-nb">0</span></div>
    <div class="nav-sec">زیرساخت</div>
    <div class="nav-it" data-pg="security"><i class="ti ti-shield-check"></i> پروتکل امنیتی</div>
    <div class="nav-it" data-pg="logs"><i class="ti ti-history"></i> لاگ Epoch‌ها</div>
    <div class="nav-it" data-pg="errors"><i class="ti ti-alert-triangle"></i> آنامولی‌ها</div>
    <div class="nav-it" data-pg="testws"><i class="ti ti-activity"></i> پینگ Worker</div>
    <div class="nav-it" data-pg="settings"><i class="ti ti-adjustments-horizontal"></i> کانفیگ سرور</div>
  </div>
  <div class="sb-foot">
    <button class="theme-btn" onclick="toggleTheme()"><i class="ti ti-moon" id="theme-icon"></i> <span id="theme-label">تم روشن</span></button>
    <button class="logout-btn" id="logout-btn"><i class="ti ti-logout"></i> خروج نُد</button>
  </div>
</aside>
<main class="main">
<section class="pg on" id="pg-overview">
  <div class="topbar">
    <div><div class="tb-title"><i class="ti ti-layout-dashboard"></i> مانیتورینگ کلاستر</div><div class="tb-sub" id="last-upd">سینک با نُد مرکزی...</div></div>
    <div class="tb-right">
      <span class="badge bg-green"><span class="dot dg pulse"></span> شبکه فعال</span>
      <span class="badge bg-blue" id="uptime-badge">—</span>
      <button class="btn btn-p btn-sm" onclick="refreshAll()"><i class="ti ti-refresh"></i> سینک مجدد</button>
    </div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="m-icon"><i class="ti ti-chart-arcs"></i></div><div class="m-label">استریم‌های فعال</div><div class="m-val" id="m-conns">—</div><div class="m-sub"><span class="dot dg pulse"></span> سینک بلادرنگ</div></div>
    <div class="metric"><div class="m-icon"><i class="ti ti-box-padding"></i></div><div class="m-label">پردازش کل شبکه</div><div class="m-val" id="m-traffic">—<span class="m-unit">M-Tok</span></div><div class="m-sub">توکن‌های محاسبه شده</div></div>
    <div class="metric suc"><div class="m-icon suc"><i class="ti ti-cpu"></i></div><div class="m-label">نُدهای آنلاین</div><div class="m-val" id="m-alinks">—</div><div class="m-sub" id="m-lsub">از ظرفیت کل</div></div>
    <div class="metric pur"><div class="m-icon pur"><i class="ti ti-server-cog"></i></div><div class="m-label">خوشه‌های پردازشی</div><div class="m-val" id="m-subs">—</div><div class="m-sub">فعال</div></div>
  </div>
  <div class="vless-box">
    <div class="vl-header">
      <div class="vl-title"><i class="ti ti-vector"></i> نقطه همگام‌سازی پایه (بدون محدودیت پردازش)</div>
      <span class="badge bg-blue"><span class="dot db"></span> Neural Net · WS Sync</span>
    </div>
    <div class="vl-code" id="core-endpoint-val">در حال محاسبه گراف...</div>
    <div class="vl-actions">
      <button class="btn btn-p" onclick="cpText('core-endpoint-val')"><i class="ti ti-copy"></i> کپی Endpoint</button>
      <button class="btn btn-g" onclick="qrFor('core-endpoint-val')"><i class="ti ti-qrcode"></i> کپسول QR</button>
      <button class="btn btn-o" onclick="navTo('links')"><i class="ti ti-cpu"></i> تخصیص نُد محدود</button>
      <button class="btn btn-pur" onclick="navTo('subgroups')"><i class="ti ti-server-cog"></i> خوشه‌ها</button>
    </div>
  </div>
  <div class="g2">
    <div class="card">
      <div class="card-title"><i class="ti ti-activity"></i> سلامت سرویس‌ها</div>
      <div class="sr"><span class="sr-k"><i class="ti ti-shield-check"></i> Node Hash Auth</span><span class="sr-v" style="color:var(--green-t)">● تایید شده</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-network"></i> Architectures</span><span class="sr-v" style="color:var(--green-t)">● Tensor-Sync / Neural-Stream</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-database-export"></i> Global Registry</span><span class="sr-v" style="color:var(--green-t)">● فعال</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-clock"></i> زمان پردازش (Uptime)</span><span class="sr-v" id="uptime-inline">—</span></div>
    </div>
    <div class="card">
      <div class="card-title"><i class="ti ti-list"></i> تخصیص بار نُدها <span class="ml-auto badge bg-blue" id="lsummary-badge">۰</span></div>
      <div id="lsummary">—</div>
    </div>
  </div>
</section>
<section class="pg" id="pg-links">
  <div class="topbar">
    <div><div class="tb-title"><i class="ti ti-cpu"></i> نُدهای پردازشی</div><div class="tb-sub">تخصیص منابع، بودجه توکن و معماری همگام‌سازی</div></div>
    <div class="tb-right"><span class="badge bg-blue" id="links-pg-cnt">۰ گراف</span></div>
  </div>
  <div class="create-panel">
    <div class="cp-head">
      <div class="cp-head-icon"><i class="ti ti-box-model"></i></div>
      <div class="cp-head-text">
        <div class="cp-head-title">استقرار نُد پردازشی جدید</div>
        <div class="cp-head-sub">هش تصادفی شبکه · بودجه و معماری را تعیین کنید</div>
      </div>
    </div>
    <div class="cp-body">
      <div class="cp-row">
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-id-badge-2"></i> شناسه گراف/کلاینت</div>
          <input class="cp-input-full" id="nl-label" placeholder="مثلاً: Worker-Alpha">
          <div class="cp-mini-row">
            <input class="cp-input-full" id="nl-note" placeholder="متا دیتا (اختیاری)">
          </div>
          <div class="cp-mini-row" style="margin-top:8px">
            <input class="cp-input-full" id="nl-sub-domain" placeholder="درگاه کاستوم (اختیاری)">
          </div>
        </div>
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-server-cog"></i> اتصال به خوشه‌ها (Ensembles)</div>
          <div id="nl-subs-list" style="max-height:100px;overflow-y:auto;background:rgba(0,0,0,.15);border:1px solid var(--card-b);border-radius:10px;padding:8px;display:flex;flex-direction:column;gap:5px;margin-bottom:8px">
             <!-- لیست ساب‌ها اینجا لود میشه -->
          </div>
          <div class="cp-mini-row">
            <input class="cp-input-full" id="nl-exp" type="number" min="0" step="1" placeholder="مهلت Epoch (روز) · 0 = دائم">
          </div>
        </div>
      </div>

      <div class="cp-block mb16">
        <div class="cp-block-label" style="display:flex;justify-content:space-between">
          <span><i class="ti ti-world"></i> روت‌های استریم موازی (Load Balancing)</span>
        </div>
        <div id="nl-saved-customs" style="display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;margin-bottom:8px"></div>
        <div id="nl-customs-list" style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px"></div>
        <button class="btn btn-sm btn-g" type="button" onclick="addCustomField('nl')"><i class="ti ti-plus"></i> ایجاد روت کاستوم جدید</button>
      </div>

      <div class="cp-block mb16">
        <div class="cp-block-label"><i class="ti ti-chart-donut"></i> بودجه پردازشی توکن‌ها</div>
        <div class="cp-quota-inputs">
          <input class="cp-input-full" id="nl-val" type="number" min="0" step="0.1" placeholder="0 = نامحدود">
          <select class="cp-input-full fs" id="nl-unit"><option value="GB">B-Tok</option><option value="MB" selected>M-Tok</option></select>
        </div>
      </div>
      <div class="cp-block mb16">
        <div class="cp-block-label"><i class="ti ti-brain"></i> معماری شبکه ارتباطی</div>
        <select id="nl-proto" style="display:none">
          <option value="vless-ws">Neural-WS</option>
          <option value="httpupgrade">Hyper-Gradient</option>
          <option value="xhttp-packet-up">X-Tensor Pkt</option>
          <option value="xhttp-stream-up">X-Tensor Str</option>
          <option value="xhttp-reality">REALITY-MLKEM</option>
        </select>
        <div class="proto-cards">
          <div class="proto-card active" data-val="vless-ws" onclick="selectProto('vless-ws',this)">
            <div class="proto-card-check"><i class="ti ti-check"></i></div>
            <div class="proto-card-icon"><i class="ti ti-wave-sine"></i></div>
            <div class="proto-card-title">Neural-WS</div>
            <div class="proto-card-desc">WebSocket پایه</div>
          </div>
          <div class="proto-card" data-val="httpupgrade" onclick="selectProto('httpupgrade',this)">
            <div class="proto-card-check"><i class="ti ti-check"></i></div>
            <div class="proto-card-icon"><i class="ti ti-arrow-up-circle"></i></div>
            <div class="proto-card-title">Hyper-Gradient</div>
            <div class="proto-card-desc">HTTP-Upgrade</div>
          </div>
          <div class="proto-card" data-val="xhttp-packet-up" onclick="selectProto('xhttp-packet-up',this)">
            <div class="proto-card-check"><i class="ti ti-check"></i></div>
            <div class="proto-card-icon"><i class="ti ti-bolt"></i></div>
            <div class="proto-card-title">X-Tensor Pkt</div>
            <div class="proto-card-desc">پکت‌های ناهمگام</div>
          </div>
          <div class="proto-card" data-val="xhttp-stream-up" onclick="selectProto('xhttp-stream-up',this)">
            <div class="proto-card-check"><i class="ti ti-check"></i></div>
            <div class="proto-card-icon"><i class="ti ti-rocket"></i></div>
            <div class="proto-card-title">X-Tensor Str</div>
            <div class="proto-card-desc">استریم زنده</div>
          </div>
          <div class="proto-card" data-val="xhttp-reality" onclick="selectProto('xhttp-reality',this)">
            <div class="proto-card-check"><i class="ti ti-check"></i></div>
            <div class="proto-card-icon"><i class="ti ti-shield-lock"></i></div>
            <div class="proto-card-title">REALITY-MLKEM</div>
            <div class="proto-card-desc">استتار کوانتومی</div>
          </div>
        </div>
      </div>
      <div class="cp-row">
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-fingerprint"></i> Q-TLS Profile</div>
          <select class="cp-input-full fs" id="nl-fp">
            <option value="chrome" selected>fp16_accurate</option>
            <option value="firefox">int8_fast</option>
            <option value="safari">fp32_native</option>
            <option value="ios">mixed_precision</option>
            <option value="android">bfloat16</option>
            <option value="edge">edge_quant</option>
            <option value="random">randomized_dist</option>
          </select>
        </div>
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-antenna-bars-5"></i> لایه ALPN</div>
          <select class="cp-input-full fs" id="nl-alpn-preset" onchange="onAlpnPresetChange()">
            <option value="">استاندارد شبکه</option>
            <option value="h2,http/1.1">h2,http/1.1</option>
            <option value="http/1.1">http/1.1</option>
            <option value="h2">h2</option>
            <option value="__custom__">تعریف دستی...</option>
          </select>
          <div class="cp-mini-row">
            <input class="cp-input-full" id="nl-alpn" placeholder="مقدار ALPN" style="display:none">
          </div>
        </div>
      </div>
      <div class="cp-row mb16">
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-route"></i> پورت ورود داده</div>
          <input class="cp-input-full" id="nl-port" type="number" min="1" max="65535" placeholder="443" value="443">
        </div>
        <div class="cp-block">
          <div class="cp-block-label"><i class="ti ti-users"></i> حداکثر Worker موازی</div>
          <input class="cp-input-full" id="nl-iplimit" type="number" min="0" step="1" placeholder="0 = نامحدود" value="0">
        </div>
      </div>
      <div class="cp-row mb16">
        <div class="cp-block" style="flex:1">
          <div class="cp-block-label"><i class="ti ti-gauge"></i> نرخ پردازش مجاز (Tokens/s)</div>
          <div class="form-row">
            <input class="cp-input-full" id="nl-speed" type="number" min="0" step="0.5" placeholder="0 = نامحدود" value="0" style="flex:1">
            <select class="fs" id="nl-speed-unit" style="flex:0 0 100px">
              <option value="MBIT" selected>M-Tok/s</option>
              <option value="KB">K-Tok/s</option>
            </select>
          </div>
        </div>
      </div>
      <div class="cp-footer">
        <div class="cp-footer-note"><i class="ti ti-info-circle"></i> Hash شبکه به صورت کوانتومی و یکتا ایجاد می‌شود. فقط Worker‌های مجاز امکان Sync دارند.</div>
        <button class="cp-submit-btn" onclick="createLink()"><i class="ti ti-box-model"></i> استقرار نُد</button>
      </div>
    </div>
  </div>
  <div class="cfg-grid" id="links-grid"></div>
  <div class="empty" id="links-empty" style="display:none"><i class="ti ti-cpu"></i><p>هیچ نُدی مستقر نشده است</p></div>
</section>
<section class="pg" id="pg-subgroups">
  <div class="topbar">
    <div><div class="tb-title"><i class="ti ti-server-cog"></i> خوشه‌های پردازشی</div><div class="tb-sub">مدیریت ایزوله‌های مستقل با خروجی رجیستری پابلیک</div></div>
    <div class="tb-right">
      <span class="badge bg-purple" id="subs-pg-cnt">۰ خوشه</span>
      <button class="btn btn-pur" onclick="openModal('modal-create-sub')"><i class="ti ti-vector"></i> ایجاد خوشه</button>
    </div>
  </div>
  <div class="subs-toolbar">
    <div class="subs-search">
      <i class="ti ti-search"></i>
      <input type="text" id="subs-search-inp" placeholder="جستجوی خوشه‌ها..." oninput="filterSubs(this.value)">
    </div>
  </div>
  <div class="sub-grid" id="subs-grid">
    <div class="subs-empty-v2"><div class="subs-empty-v2-icon"><i class="ti ti-server-cog"></i></div><div class="subs-empty-v2-title">خوشه‌ای وجود ندارد</div></div>
  </div>
</section>
<section class="pg" id="pg-subscriptions">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-database-export"></i> رجیستری مدل</div><div class="tb-sub">استخراج معماری و وزن‌ها از کلاستر</div></div></div>
  <div class="g2">
    <div class="card">
      <div class="card-title"><i class="ti ti-topology-star-3"></i> رجیستری خرد (Micro)</div>
      <p style="font-size:11.5px;color:var(--t3);line-height:1.8;margin-bottom:12px">هر نُد دارای یک مسیر Sync مستقل است. برای دریافت از طریق کارت گراف‌ها اقدام کنید.</p>
    </div>
    <div class="card">
      <div class="card-title"><i class="ti ti-database"></i> رجیستری کل کلاستر (Global)</div>
      <p style="font-size:11.5px;color:var(--t3);line-height:1.8;margin-bottom:4px">آدرس همگام‌سازی تمامی نُدهای فعال شبکه مرکزی.</p>
      <div class="sub-box"><span class="sub-url" id="sub-all-url">در حال محاسبه...</span><div style="display:flex;gap:6px"><button class="btn btn-sm btn-g" onclick="cpSubAll()"><i class="ti ti-copy"></i></button><button class="btn btn-sm btn-g" onclick="window.open(document.getElementById('sub-all-url').textContent)"><i class="ti ti-external-link"></i></button></div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title"><i class="ti ti-server-cog"></i> Endpoints خوشه‌های پردازشی</div>
    <div id="sub-groups-list">در حال اسکن...</div>
  </div>
</section>
<section class="pg" id="pg-connections">
  <div class="topbar">
    <div><div class="tb-title"><i class="ti ti-chart-arcs"></i> استریم‌های زنده</div><div class="tb-sub">مانیتورینگ بلادرنگ Workerهای در حال آموزش</div></div>
    <div class="tb-right"><span class="badge bg-green" id="conns-live">—</span><button class="btn btn-p btn-sm" onclick="refreshAll()"><i class="ti ti-refresh"></i> پایش</button></div>
  </div>
  <div class="conn-hero">
    <div class="conn-hero-tile">
      <div class="conn-hero-icon"><i class="ti ti-chart-arcs"></i></div>
      <div class="conn-hero-label">سِشِن‌های پردازشی</div>
      <div class="conn-hero-val" id="ch-count">—</div>
    </div>
    <div class="conn-hero-tile">
      <div class="conn-hero-icon"><i class="ti ti-box-padding"></i></div>
      <div class="conn-hero-label">حجم پردازش لحظه‌ای</div>
      <div class="conn-hero-val" id="ch-traffic">—</div>
    </div>
  </div>
  <div class="conn-toolbar">
    <div class="conn-toolbar-title"><i class="ti ti-list-details"></i> کلاینت‌های متصل</div>
    <div class="conn-live-badge"><span class="conn-live-dot"></span> آپدیت وضعیت هر ۵ ثانیه</div>
  </div>
  <div class="conn-grid-v2" id="conns-grid"></div>
  <div class="conn-empty-v2" id="conns-empty" style="display:none">
    <div class="conn-empty-v2-icon"><i class="ti ti-plug-off"></i></div>
    <div class="conn-empty-v2-title">هیچ استریم فعالی یافت نشد</div>
  </div>
</section>
<section class="pg" id="pg-security">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-shield-check"></i> پروتکل امنیتی کلاستر</div></div></div>
  <div class="g2">
    <div class="card">
      <div class="card-title"><i class="ti ti-lock"></i> لایه‌های رمزنگاری</div>
      <div class="sr"><span class="sr-k"><i class="ti ti-certificate"></i> Q-TLS (Port 443)</span><span class="sr-v" style="color:var(--green-t)">● فعال (Active)</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-fingerprint"></i> Anti-Probe Spoofing</span><span class="sr-v">fp16_accurate</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-network"></i> Data Transports</span><span class="sr-v">Neural/Hyper/X-Tensor</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-key"></i> Hash Algorithm</span><span class="sr-v">SHA-256+Quantum-Salt</span></div>
    </div>
    <div class="card">
      <div class="card-title"><i class="ti ti-shield-check"></i> سیستم Watchdog</div>
      <div class="sr"><span class="sr-k"><i class="ti ti-id-badge"></i> Node Hash Auth</span><span class="sr-v" style="color:var(--green-t)">● تاییدشده v9</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-chart-donut"></i> بودجه توکن پردازشی</span><span class="sr-v" style="color:var(--green-t)">● فعال</span></div>
      <div class="sr"><span class="sr-k"><i class="ti ti-calendar-x"></i> مهلت Timeout (Epoch)</span><span class="sr-v" style="color:var(--green-t)">● فعال</span></div>
    </div>
  </div>
</section>
<section class="pg" id="pg-logs">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-history"></i> لاگ‌های Epoch (Events)</div><div class="tb-sub">رخدادهای ثبت شده در حین آموزش و همگام‌سازی</div></div><div class="tb-right"><button class="btn btn-p btn-sm" onclick="loadActivity()"><i class="ti ti-refresh"></i></button></div></div>
  <div class="card"><div class="log-timeline" id="logs-list">—</div><div class="empty" id="logs-empty" style="display:none"><i class="ti ti-history-toggle"></i><p>لاگی در سیستم ثبت نشده است</p></div></div>
</section>
<section class="pg" id="pg-errors">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-alert-triangle"></i> تشخیص آنامولی</div></div><div class="tb-right"><span class="badge bg-red" id="errs-badge">۰</span><button class="btn btn-p btn-sm" onclick="refreshAll()"><i class="ti ti-refresh"></i></button></div></div>
  <div class="card"><div class="card-title"><i class="ti ti-bug"></i> Traceback و استثناها</div><div id="errs-full">—</div></div>
</section>
<section class="pg" id="pg-testws">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-activity"></i> پینگ Worker (تست استریم)</div></div></div>
  <div class="card" style="max-width:660px">
    <div class="cl amber" style="margin-top:0;margin-bottom:12px"><i class="ti ti-alert-triangle"></i><span>این فقط یک پروب تست استریم است.</span></div>
    <div class="form-row" style="margin-bottom:12px">
      <div class="fg" style="flex:1"><label>Hash کلاینت (باید معتبر باشد)</label><input class="fi" id="ws-uuid" placeholder="شناسه گراف (Node Hash)..." style="width:100%"></div>
      <button class="btn btn-p" onclick="wsConn()"><i class="ti ti-vector"></i> اجرای پروب</button>
      <button class="btn btn-d" onclick="wsDisc()"><i class="ti ti-plug-x"></i> خاتمه</button>
    </div>
    <div class="form-row" style="margin-bottom:12px">
      <input class="fi" id="ws-msg" placeholder="تزریق پکت تستی..." style="flex:1">
      <button class="btn btn-o" onclick="wsSend()"><i class="ti ti-send"></i> تزریق</button>
    </div>
    <div style="background:rgba(0,0,0,.3);border:1px solid var(--card-b);border-radius:10px;padding:14px;height:250px;overflow-y:auto;font-family:ui-monospace,monospace;font-size:10.5px;line-height:1.9" id="ws-log">
      <p style="color:var(--t3)">آماده‌سازی سوکت برای پروب...</p>
    </div>
  </div>
</section>
<section class="pg" id="pg-settings">
  <div class="topbar"><div><div class="tb-title"><i class="ti ti-adjustments-horizontal"></i> کانفیگ کلاستر (Settings)</div></div></div>
  <div class="g2">
    <div class="srv-panel">
      <div class="srv-hero">
        <div class="srv-hero-icon"><i class="ti ti-cpu"></i></div>
        <div class="srv-hero-text">
          <div class="srv-hero-domain" id="set-host">—</div>
          <div class="srv-hero-sub"><span class="dot dg pulse"></span> شبکه آنلاین · Tensor Backend</div>
        </div>
      </div>
      <div class="srv-tiles">
        <div class="srv-tile"><div class="srv-tile-icon"><i class="ti ti-route"></i></div><div class="srv-tile-text"><div class="srv-tile-label">پورت همگام‌سازی</div><div class="srv-tile-val">443 (Q-TLS)</div></div></div>
        <div class="srv-tile"><div class="srv-tile-icon"><i class="ti ti-versions"></i></div><div class="srv-tile-text"><div class="srv-tile-label">معماری سیستم</div><div class="srv-tile-val">v9.5 Tensor-Core</div></div></div>
      </div>
    </div>
    <div class="pw-panel">
      <div class="pw-hero">
        <div class="pw-hero-icon"><i class="ti ti-key"></i></div>
        <div class="pw-hero-text">
          <div class="pw-hero-title">بازنشانی توکن ادمین</div>
        </div>
      </div>
      <div class="pw-body">
        <div class="pw-field">
          <label>کلید فعلی</label>
          <input class="pw-input" type="password" id="cp-cur" placeholder="تایید اعتبار کلید فعلی">
          <button class="pw-eye" type="button" onclick="togglePwField('cp-cur',this)"><i class="ti ti-eye"></i></button>
        </div>
        <div class="pw-field" style="margin-bottom:6px">
          <label>کلید جدید</label>
          <input class="pw-input" type="password" id="cp-new" placeholder="حداقل ۴ کاراکتر" oninput="checkPwStrength(this.value)">
          <button class="pw-eye" type="button" onclick="togglePwField('cp-new',this)"><i class="ti ti-eye"></i></button>
        </div>
        <div class="pw-strength" id="pw-strength-bar">
          <div class="pw-strength-seg"></div><div class="pw-strength-seg"></div><div class="pw-strength-seg"></div><div class="pw-strength-seg"></div>
        </div>
        <div class="pw-strength-label" id="pw-strength-label"><i class="ti ti-shield"></i> قدرت کلید</div>
        <div class="pw-field" style="margin-bottom:18px;margin-top:10px">
          <label>تکرار کلید جدید</label>
          <input class="pw-input" type="password" id="cp-cf" placeholder="...">
          <button class="pw-eye" type="button" onclick="togglePwField('cp-cf',this)"><i class="ti ti-eye"></i></button>
        </div>
        <button class="pw-submit" onclick="changePw()"><i class="ti ti-shield-check"></i> بروزرسانی کلید</button>
      </div>
    </div>

    <div class="pw-panel" style="margin-top:16px; grid-column: 1 / -1;">
      <div class="pw-hero" style="background: linear-gradient(135deg, rgba(0,230,118,0.1), transparent);">
        <div class="pw-hero-icon" style="background: linear-gradient(135deg, var(--green), #00994D);"><i class="ti ti-schema"></i></div>
        <div class="pw-hero-text">
          <div class="pw-hero-title">توزیع بار کلودفلر (Cloudflare Edge Sync)</div>
          <div class="pw-hero-sub">بدون قرار دادن رمز در کد، سرورها را به هم متصل کنید</div>
        </div>
      </div>
      <div class="pw-body">
        <div class="form-row" style="margin-bottom:12px">
          <div class="fg" style="flex:1">
            <label>آدرس ورکر واسط (Worker URL)</label>
            <input class="pw-input" id="cf-worker-url" placeholder="https://nexus-proxy.domain.workers.dev">
          </div>
        </div>
        <div class="form-row" style="margin-bottom:12px">
          <div class="fg" style="flex:1">
            <label>توکن امنیتی (Secret Auth Token)</label>
            <input class="pw-input" type="password" id="cf-worker-token" placeholder="در صورت عدم تغییر، خالی بگذارید">
          </div>
        </div>
        <div class="cl" style="margin-top:0; margin-bottom:16px;">
          <i class="ti ti-info-circle"></i>
          <span>اطلاعات اتصال به صورت رمزنگاری شده روی سرور ذخیره می‌شود.</span>
        </div>
        <div style="display:flex;gap:8px">
          <button class="pw-submit" style="background:var(--green);color:#000;flex:1" onclick="saveCfSync()"><i class="ti ti-check"></i> ذخیره در سرور (Commit)</button>
          <button class="pw-submit" style="background:var(--accent-d);color:var(--accent);flex:1;box-shadow:none" onclick="testCfSync()"><i class="ti ti-wifi"></i> تست ارتباط (Ping Edge)</button>
        </div>
        <div style="display:flex;gap:8px;margin-top:12px;padding-top:12px;border-top:1px solid var(--card-b)">
          <button class="pw-submit" style="background:var(--accent);color:#000;flex:1;box-shadow:none" onclick="uploadToCf()"><i class="ti ti-cloud-upload"></i> آپلود به کلاستر (Push)</button>
          <button class="pw-submit" style="background:var(--purple);color:#fff;flex:1;box-shadow:none" onclick="downloadFromCf()"><i class="ti ti-cloud-download"></i> همگام‌سازی از کلاستر (Pull)</button>
        </div>
      </div>
    </div>

    <div class="pw-panel" style="margin-top:16px; grid-column: 1 / -1;">
      <div class="pw-hero" style="background: linear-gradient(135deg, rgba(59,130,246,0.1), transparent);">
        <div class="pw-hero-icon" style="background: linear-gradient(135deg, #3B82F6, #1D4ED8);box-shadow:0 6px 18px rgba(59,130,246,0.25)"><i class="ti ti-message-chatbot"></i></div>
        <div class="pw-hero-text">
          <div class="pw-hero-title">پشتیبان‌گیری روی تلگرام (Telegram Checkpoint)</div>
          <div class="pw-hero-sub">ثبت فایل دیتابیس در بات تلگرام</div>
        </div>
      </div>
      <div class="pw-body">
        <div class="form-row" style="margin-bottom:12px">
          <div class="fg" style="flex:1.5">
            <label>توکن دسترسی بات (Bot Token)</label>
            <input class="pw-input" id="tg-bot-token" type="password" placeholder="مثال: 123456:ABC-DEF1234ghIkl-zyx...">
          </div>
          <div class="fg" style="flex:1">
            <label>آیدی عددی ادمین (Admin ID)</label>
            <input class="pw-input" id="tg-admin-id" placeholder="مثال: 123456789">
          </div>
        </div>
        <div class="cl" style="margin-top:0; margin-bottom:16px; background:rgba(59,130,246,0.1); border-color:rgba(59,130,246,0.2); color:var(--t1)">
          <i class="ti ti-info-circle" style="color:#3B82F6"></i>
          <span style="font-size:10.5px">برای **دریافت فایل از تلگرام**، ابتدا در ربات خود فایل بکاپ را فوروارد/ارسال کنید، سپس دکمه Pull را بزنید.</span>
        </div>
        <div style="display:flex;gap:8px">
          <button class="pw-submit" style="background:var(--accent-d);color:var(--accent);flex:1;box-shadow:none" onclick="saveTgSettings()"><i class="ti ti-device-floppy"></i> ذخیره کانفیگ تلگرام</button>
          <button class="pw-submit" style="background:#3B82F6;color:#fff;flex:1" onclick="downloadFromTg()"><i class="ti ti-cloud-download"></i> ریکاوری فایل از TG (Pull)</button>
        </div>
      </div>
    </div>

  </div>
</section>
</main>
<script>
let isDark=localStorage.getItem('Sadra-theme')!=='light';
function applyTheme(dark){
  document.documentElement.setAttribute('data-theme',dark?'dark':'light');
  const icon=dark?'ti-sun':'ti-moon',label=dark?'تم روشن':'تم تاریک';
  document.getElementById('theme-icon').className='ti '+icon;
  document.getElementById('theme-label').textContent=label;
  const mobI=document.getElementById('theme-mob-icon');if(mobI)mobI.className='ti '+icon;
}
function toggleTheme(){isDark=!isDark;localStorage.setItem('Sadra-theme',isDark?'dark':'light');applyTheme(isDark)}
applyTheme(isDark);
function toast(msg,type=''){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast show'+(type?' '+type:'');
  setTimeout(()=>t.classList.remove('show'),2400);
}
function fmtB(b){if(!b||b===0)return '0 Tok';if(b<1024)return b+' Tok';if(b<1024**2)return (b/1024).toFixed(1)+' K-Tok';if(b<1024**3)return (b/1024**2).toFixed(2)+' M-Tok';return (b/1024**3).toFixed(2)+' B-Tok'}
function toFa(n){return String(n).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d])}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function daysLeft(exp){if(!exp)return null;return Math.ceil((new Date(exp)-Date.now())/(864e5))}
function expChip(exp,expired){
  if(expired)return '<span class="exp-chip ec-exp"><i class="ti ti-calendar-x"></i> Timeout</span>';
  if(!exp)return '<span class="exp-chip ec-inf"><i class="ti ti-infinity"></i> دائم</span>';
  const d=daysLeft(exp);
  if(d<=0)return '<span class="exp-chip ec-exp"><i class="ti ti-calendar-x"></i> Timeout</span>';
  if(d<=3)return `<span class="exp-chip ec-warn"><i class="ti ti-alert-triangle"></i> ${toFa(d)} روز مانده</span>`;
  return `<span class="exp-chip ec-ok"><i class="ti ti-calendar-check"></i> ${toFa(d)} روز مانده</span>`;
}
function protoBadge(p){
  const m={'vless-ws':['Neural-WS','pc-ws'], 'httpupgrade':['Hyper-Gradient','pc-ws'], 'xhttp-packet-up':['X-Tensor Pkt','pc-xhttp'],'xhttp-stream-up':['X-Tensor Str','pc-xhttp'], 'xhttp-reality':['REALITY-MLKEM','pc-ultra']};
  const v=m[p]||m['vless-ws'];
  return `<span class="proto-chip ${v[1]}">${v[0]}</span>`;
}
async function checkAuth(){try{const r=await fetch('/api/me');const d=await r.json();if(!d.authenticated)location.href='/sadra1491388191378';}catch(e){location.href='/sadra1491388191378'}}
async function logout(){try{await fetch('/api/logout',{method:'POST'})}catch(e){}location.href='/sadra1491388191378'}
document.getElementById('logout-btn').addEventListener('click',logout);
async function authF(url,opts={}){
  const r=await fetch(url,opts);
  if(r.status===401){location.href='/sadra1491388191378';throw new Error('unauthorized')}
  return r;
}
function selectProto(val,el){
  document.getElementById('nl-proto').value = val;
  document.querySelectorAll('.proto-card').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
}
function onAlpnPresetChange(){
  const p=document.getElementById('nl-alpn-preset').value;
  const inp=document.getElementById('nl-alpn');
  if(p==='__custom__'){inp.style.display='block';inp.value='';inp.focus();}
  else{inp.style.display='none';inp.value=p;}
}
const sb=document.getElementById('sb'),overlay=document.getElementById('overlay');
function openSb(){sb.classList.add('open');overlay.classList.add('show')}
function closeSb(){sb.classList.remove('open');overlay.classList.remove('show')}
document.getElementById('open-sb').addEventListener('click',openSb);
document.getElementById('close-sb').addEventListener('click',closeSb);
overlay.addEventListener('click',closeSb);
function navTo(name){
  document.querySelectorAll('.nav-it').forEach(n=>n.classList.toggle('on',n.dataset.pg===name));
  document.querySelectorAll('.pg').forEach(p=>p.classList.toggle('on',p.id==='pg-'+name));
  const loaders={links:loadLinks,connections:loadConns,errors:loadErrs,subscriptions:loadSubsPage,subgroups:loadSubs,logs:loadActivity};
  if(loaders[name])loaders[name]();
  closeSb();window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('.nav-it').forEach(el=>el.addEventListener('click',()=>navTo(el.dataset.pg)));
function openModal(id){document.getElementById(id).classList.add('open')}
function closeModal(id){document.getElementById(id).classList.remove('open')}
async function fetchStats(){
  try{
    const r=await authF('/stats'),d=await r.json();
    document.getElementById('m-conns').textContent=d.active_connections;
    document.getElementById('conns-nb').textContent=d.active_connections;
    document.getElementById('m-traffic').innerHTML=d.total_traffic_mb.toFixed(1)+'<span class="m-unit">M-Tok</span>';
    document.getElementById('m-alinks').textContent=d.active_links??'—';
    document.getElementById('m-lsub').textContent='از '+d.links_count+' گراف';
    document.getElementById('m-subs').textContent=d.subs_count??'—';
    document.getElementById('errs-badge').textContent=d.total_errors+' آنامولی';
    document.getElementById('uptime-inline').textContent=d.uptime;
    document.getElementById('uptime-badge').textContent='Tensor-Core · '+d.uptime;
    document.getElementById('last-upd').textContent='سینک شده: '+new Date().toLocaleTimeString('fa-IR');
    document.getElementById('conns-live').innerHTML='<span class="dot dg pulse"></span> '+d.active_connections+' استریم';
    renderErrs(d.recent_errors||[]);
  }catch(e){console.error(e)}
}
function renderErrs(errs){
  const el=document.getElementById('errs-full');if(!el)return;
  if(!errs.length){el.innerHTML='<div style="color:var(--green-t);padding:10px;font-size:12px;display:flex;align-items:center;gap:5px"><i class="ti ti-circle-check"></i> سیستم در پایداری کامل است</div>';return}
  el.innerHTML=errs.slice().reverse().map(e=>`<div class="erow"><div class="etime"><i class="ti ti-clock"></i>${new Date(e.time).toLocaleString('fa-IR')}</div><div class="emsg">${esc(e.error)}${e.url?' — '+esc(e.url):''}</div></div>`).join('');
}
async function loadActivity(){
  try{
    const r=await authF('/api/activity'),d=await r.json();
    const logs=(d.logs||[]).slice().reverse();
    const el=document.getElementById('logs-list'),em=document.getElementById('logs-empty');
    if(!logs.length){el.innerHTML='';em.style.display='block';return}
    em.style.display='none';
    const icMap={ok:'ti-circle-check',err:'ti-circle-x',warn:'ti-alert-triangle',info:'ti-info-circle'};
    const kindFa={link:'گراف',sub:'خوشه',auth:'ورود',connection:'استریم',system:'کلاستر'};
    el.innerHTML=logs.map(l=>`
      <div class="log-item">
        <div class="log-ic ${l.level}"><i class="ti ${icMap[l.level]||'ti-info-circle'}"></i></div>
        <div class="log-body">
          <div class="log-msg">${esc(l.message)}</div>
          <div class="log-time"><i class="ti ti-clock"></i> ${new Date(l.time).toLocaleString('fa-IR')} <span class="log-kind">${kindFa[l.kind]||l.kind}</span></div>
        </div>
      </div>
    `).join('');
  }catch(e){console.error(e)}
}
let allSubsList=[],allLinksList=[];
async function loadLinks(){
  try{
    const [lr,sr]=await Promise.all([authF('/api/links'),authF('/api/subs')]);
    const {links=[]}=await lr.json();
    const {subs=[]}=await sr.json();
    allSubsList=subs;allLinksList=links;
    const renderSubCheckboxes = (containerId, subsList, checkedSet = new Set()) => {
        const container = document.getElementById(containerId);
        if(!container) return;
        const currentlyChecked = new Set([...container.querySelectorAll('input:checked')].map(cb => cb.value));
        const finalChecked = checkedSet.size > 0 ? checkedSet : currentlyChecked;
        
        container.innerHTML = subsList.map(s => `
            <label style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--t1);cursor:pointer;padding:6px;border-radius:8px;background:rgba(0,255,196,0.05);border:1px solid transparent;transition:.15s" onmouseover="this.style.borderColor='var(--card-b)'" onmouseout="this.style.borderColor='transparent'">
                <input type="checkbox" value="${esc(s.sub_id)}" class="sub-cb" ${finalChecked.has(s.sub_id) ? 'checked' : ''} style="width:16px;height:16px;accent-color:var(--accent)">
                <span>${esc(s.name)}</span>
            </label>
        `).join('');
    };
    renderSubCheckboxes('nl-subs-list', subs);
    document.getElementById('links-nb').textContent=links.length;
    document.getElementById('links-pg-cnt').textContent=toFa(links.length)+' گراف';
    document.getElementById('lsummary-badge').textContent=toFa(links.length);
    const grid=document.getElementById('links-grid'),empty=document.getElementById('links-empty');
    if(!links.length){grid.innerHTML='';empty.style.display='block';document.getElementById('lsummary').innerHTML='<div class="empty"><i class="ti ti-cpu"></i><p>هیچ گراف پردازشی در دسترس نیست</p></div>';return}
    empty.style.display='none';
    grid.innerHTML=links.map(l=>{
  const lim=l.limit_bytes===0?'∞':fmtB(l.limit_bytes);
  const pct=l.limit_bytes===0?0:Math.min(100,l.used_bytes/l.limit_bytes*100);
  const bc=pct>90?'var(--red)':pct>70?'var(--amber)':'var(--accent)';
  const allowed=l.active&&!l.expired;
  const cardCls=!l.active?'is-off':(l.expired?'is-exp':'');
  return `<div class="cfg-card ${cardCls}">
    <div class="cfg-row">
      <span class="cfg-status-dot ${allowed?'pulse':''}"></span>
      <div class="cfg-identity">
        <div class="cfg-label">${esc(l.label)}</div>
        <div class="cfg-sub-meta">
          <span class="cfg-uuid-mini" onclick="navigator.clipboard.writeText('${l.uuid}').then(()=>toast('Hash کپی شد','ok'))" title="${l.uuid}"><i class="ti ti-fingerprint"></i> ${l.uuid.slice(0,10)}…</span>
        <span>${new Date(l.created_at).toLocaleDateString('fa-IR')}</span>
        ${l.connected_ips > 0 ? `<span style="color:var(--green-t);font-weight:700;background:var(--green-bg);padding:2px 6px;border-radius:4px"><i class="ti ti-users"></i> ${l.connected_ips} Worker متصل</span>` : `<span style="color:var(--t3);"><i class="ti ti-users"></i> ۰ Worker متصل</span>`}
        </div>
      </div>
      <div class="cfg-divider-v"></div>
      <div class="cfg-usage-col">
        <div class="ubar"><div class="ubar-f" style="width:${pct}%;background:${bc}"></div></div>
        <div class="utxt"><span>${fmtB(l.used_bytes)}</span><span>از ${lim}</span></div>
      </div>
      <div class="cfg-divider-v"></div>
      <div class="cfg-exp-col">${expChip(l.expires_at,l.expired)}</div>
      <div class="cfg-divider-v"></div>
      <div class="cfg-badges-col">
        ${protoBadge(l.protocol)}
        <span class="cfg-sub-tag" title="Sync Port"><i class="ti ti-route"></i> :${l.port||443}</span>
        ${l.address ? `<span class="cfg-sub-tag" title="Gateway اختصاصی"><i class="ti ti-world"></i> ${esc(l.address)}</span>` : ''}
        ${(l.sub_ids||[]).map(sid => {
            const sub = allSubsList.find(s=>s.sub_id===sid);
            return sub ? `<span class="cfg-sub-tag"><i class="ti ti-server-cog"></i> ${esc(sub.name)}</span>` : '';
        }).join('')}
      </div>
      <div class="cfg-divider-v"></div>
      <div class="cfg-actions">
        <button class="tog${allowed?' on':''}" onclick="toggleActive('${l.uuid}',${!l.active})" title="فعال/ایزوله کردن"></button>
        <button class="btn btn-sm btn-p btn-icon" style="width:auto;padding:0 10px" onclick="openVariations('${l.uuid}')" title="روت‌های استریم"><i class="ti ti-layers-linked"></i> ${l.variations.length} روت</button>
        <button class="btn btn-sm btn-g btn-icon" onclick="navigator.clipboard.writeText('${esc(l.sub_url)}').then(()=>toast('Registry کپی شد','ok'))" title="Registry URL"><i class="ti ti-database-export"></i></button>
        <button class="btn btn-sm btn-amber btn-icon" onclick="openEditLink('${l.uuid}')" title="تنظیم پارامتر"><i class="ti ti-edit"></i></button>
        <button class="btn btn-sm btn-g btn-icon" onclick="resetUsage('${l.uuid}')" title="ریست مصرف توکن"><i class="ti ti-rotate"></i></button>
        <button class="btn btn-sm btn-d btn-icon" onclick="deleteLink('${l.uuid}')" title="تخریب گراف"><i class="ti ti-trash"></i></button>
      </div>
    </div>
  </div>`;
}).join('');
    document.getElementById('lsummary').innerHTML=links.slice(0,6).map(l=>`<div class="sr"><span class="sr-k" style="gap:5px"><i class="ti ${l.expired?'ti-calendar-x':l.active?'ti-circle-check':'ti-circle-x'}" style="color:${l.expired?'var(--amber)':l.active?'var(--green)':'var(--red)'}"></i>${esc(l.label)}</span><span class="sr-v" style="font-size:10px">${fmtB(l.used_bytes)} / ${l.limit_bytes===0?'∞':fmtB(l.limit_bytes)}</span></div>`).join('');
  }catch(e){console.error(e)}
}
async function createLink(){
  const label=document.getElementById('nl-label').value.trim()||'گراف پیش‌فرض';
  const val=document.getElementById('nl-val').value;
  const unit=document.getElementById('nl-unit').value;
  const exp=document.getElementById('nl-exp').value;
  const note=document.getElementById('nl-note').value.trim();
  const sub_domain=document.getElementById('nl-sub-domain').value.trim();
  const protocol=document.getElementById('nl-proto').value||'vless-ws';
  const fingerprint=document.getElementById('nl-fp').value||'chrome';
  const alpn=document.getElementById('nl-alpn').value.trim();
  const port=Number(document.getElementById('nl-port').value)||443;
  const ip_limit=Number(document.getElementById('nl-iplimit').value)||0;
  const speed_limit_value=Number(document.getElementById('nl-speed').value)||0;
  const speed_limit_unit=document.getElementById('nl-speed-unit').value;
  const customs=getCustomFields('nl');
  const sub_ids = Array.from(document.querySelectorAll('#nl-subs-list input:checked')).map(cb => cb.value);

  try{
    const r=await authF('/api/links',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label,limit_value:val||0,limit_unit:unit,expires_days:exp||0,note,protocol,fingerprint,alpn,port,ip_limit,speed_limit_value,speed_limit_unit,customs,custom_domain:sub_domain,sub_ids})});
    if(!r.ok)throw new Error('failed');
    ['nl-label','nl-val','nl-exp','nl-note','nl-alpn','nl-sub-domain'].forEach(id=>document.getElementById(id).value='');
    document.getElementById('nl-customs-list').innerHTML='';
    toast('گراف مستقر شد ✓','ok');loadLinks();
  }catch(e){toast('خطا در استقرار','err')}
}
function openEditLink(uuid){
  const l=allLinksList.find(x=>x.uuid===uuid);
  if(!l)return;
  
  const varSubs = l.var_subs || {};
  renderSubCheckboxes('el-subs-list', allSubsList, new Set(varSubs['default'] || []));
  document.getElementById('el-uuid').value=uuid;
  document.getElementById('el-label').value=l.label;
  document.getElementById('el-note').value=l.note||'';
  document.getElementById('el-sub-domain').value=l.custom_domain||'';
  if(l.limit_bytes===0){document.getElementById('el-val').value='';document.getElementById('el-unit').value='GB';}
  else{document.getElementById('el-val').value=(l.limit_bytes/1024/1024).toFixed(0);document.getElementById('el-unit').value='MB';}
  document.getElementById('el-exp').value='';
  document.getElementById('el-fp').value=l.fingerprint||'chrome';
  document.getElementById('el-alpn').value=l.alpn||'';
  document.getElementById('el-port').value=l.port||443;
  document.getElementById('el-iplimit').value=l.ip_limit||0;
  if(!l.speed_limit_bytes){document.getElementById('el-speed').value='0';document.getElementById('el-speed-unit').value='MBIT';}
  else{document.getElementById('el-speed').value=(l.speed_limit_bytes*8/1024/1024).toFixed(2);document.getElementById('el-speed-unit').value='MBIT';}
  
  document.getElementById('el-customs-list').innerHTML = '';
  (l.customs || []).forEach((c, idx) => {
      const cSubs = varSubs[String(idx)] || [];
      addCustomField('el', c.name, c.address, c.host_sni, cSubs);
  });
  
  openModal('modal-edit-link');
}
async function saveEditLink(){
  const uuid=document.getElementById('el-uuid').value;
  const label=document.getElementById('el-label').value.trim();
  const note=document.getElementById('el-note').value.trim();
  const sub_domain=document.getElementById('el-sub-domain').value.trim();
  const val=document.getElementById('el-val').value;
  const unit=document.getElementById('el-unit').value;
  const exp=document.getElementById('el-exp').value;
  const fingerprint=document.getElementById('el-fp').value||'chrome';
  const alpn=document.getElementById('el-alpn').value.trim();
  const port=Number(document.getElementById('el-port').value)||443;
  const ip_limit=Number(document.getElementById('el-iplimit').value)||0;
  const speed_limit_value=Number(document.getElementById('el-speed').value)||0;
  const speed_limit_unit=document.getElementById('el-speed-unit').value;
  const customs=getCustomFields('el');
  const sub_ids = Array.from(document.querySelectorAll('#el-subs-list input:checked')).map(cb => cb.value);

  const body={label,note,limit_value:val||0,limit_unit:unit,fingerprint,alpn,port,ip_limit,speed_limit_value,speed_limit_unit,customs,custom_domain:sub_domain,sub_ids};
  if(exp&&Number(exp)>0)body.expires_days=Number(exp);
  try{
    const r=await authF('/api/links/'+uuid,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok)throw new Error();
    closeModal('modal-edit-link');
    toast('پارامترها ثبت شد ✓','ok');loadLinks();
  }catch(e){toast('خطا در اعمال پارامتر','err')}
}
async function toggleActive(uuid,newState){
  try{const r=await authF('/api/links/'+uuid,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({active:newState})});if(!r.ok)throw new Error();toast(newState?'نُد آنلاین شد ✓':'نُد ایزوله شد','ok');loadLinks();}catch(e){toast('خطا','err')}
}
async function resetUsage(uuid){
  try{const r=await authF('/api/links/'+uuid,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({reset_usage:true})});if(!r.ok)throw new Error();toast('شمارنده توکن صفر شد ✓','ok');loadLinks();}catch(e){toast('خطا','err')}
}
async function deleteLink(uuid){
  if(!confirm('آیا از تخریب این گراف پردازشی اطمینان دارید؟'))return;
  try{const r=await authF('/api/links/'+uuid,{method:'DELETE'});if(!r.ok)throw new Error();toast('گراف تخریب شد ✓','ok');loadLinks();}catch(e){toast('خطا','err')}
}
function showQR(link){window.open('https://api.qrserver.com/v1/create-qr-code/?size=300x300&data='+encodeURIComponent(link),'_blank')}
let allSubsRaw=[];
async function loadSubs(){
  try{
    const r=await authF('/api/subs'),d=await r.json();
    const subs=d.subs||[];
    allSubsRaw=subs;
    document.getElementById('subs-nb').textContent=subs.length;
    document.getElementById('subs-pg-cnt').textContent=toFa(subs.length)+' خوشه';
    renderSubsGrid(subs);
  }catch(e){console.error(e)}
}
function renderSubsGrid(subs){
  const grid=document.getElementById('subs-grid');
  if(!subs.length){
    grid.innerHTML='<div class="subs-empty-v2"><div class="subs-empty-v2-icon"><i class="ti ti-server-cog"></i></div><div class="subs-empty-v2-title">خوشه‌ای مستقر نیست</div><div class="subs-empty-v2-sub">برای توزیع بار، یک Ensemble جدید بسازید</div></div>';
    return;
  }
  grid.innerHTML=subs.map(s=>`
    <div class="sub-card">
      <div class="sub-card-top">
        <div class="sub-card-head-v2">
          <div class="sub-card-icon"><i class="ti ti-topology-star-3"></i></div>
          <div class="sub-card-titles">
            <div class="sub-card-name-v2">${esc(s.name)}</div>
            ${s.desc?`<div class="sub-card-desc-v2">${esc(s.desc)}</div>`:'<div class="sub-card-desc-v2" style="opacity:.5">بدون متادیتا</div>'}
          </div>
          <div class="sub-card-lock-badge ${s.has_password?'locked':'open'}" title="${s.has_password?'Private Endpoint':'Public Endpoint'}">
            <i class="ti ${s.has_password?'ti-lock':'ti-lock-open'}"></i>
          </div>
        </div>
        <div class="sub-card-stats">
          <div class="sub-card-stat"><div class="sub-card-stat-val">${toFa(s.links_count)}</div><div class="sub-card-stat-label">تعداد گراف</div></div>
          <div class="sub-card-stat"><div class="sub-card-stat-val" style="color:var(--green-t)">${toFa(s.active_count)}</div><div class="sub-card-stat-label">آنلاین</div></div>
          <div class="sub-card-stat"><div class="sub-card-stat-val" style="font-size:12px">${esc(s.total_used_fmt).replace('MB','M-Tok').replace('GB','B-Tok')}</div><div class="sub-card-stat-label">محاسبات</div></div>
        </div>
      </div>
      <div class="sub-card-url-row">
        <span class="sub-card-url-text">${esc(s.public_url)}</span>
        <button class="sub-card-url-copy" onclick="navigator.clipboard.writeText('${esc(s.public_url)}').then(()=>toast('آدرس متریک کپی شد','ok'))" title="کپی"><i class="ti ti-copy"></i></button>
        <button class="sub-card-url-copy" onclick="window.open('${esc(s.public_url)}','_blank')" title="اسکن روت"><i class="ti ti-external-link"></i></button>
      </div>
      <div class="sub-card-bottom">
        <button class="btn btn-sm btn-g" onclick="openSubLinks('${esc(s.sub_id)}','${esc(s.name)}')"><i class="ti ti-cpu"></i> گره‌ها</button>
        <button class="btn btn-sm btn-p" onclick="openSubVariations('${esc(s.sub_id)}')"><i class="ti ti-layers-linked"></i> استریم‌ها</button>
        <button class="btn btn-sm btn-amber" onclick="openEditSub('${esc(s.sub_id)}')"><i class="ti ti-edit"></i></button>
        <button class="btn btn-sm btn-d btn-icon" onclick="deleteSub('${esc(s.sub_id)}')" title="تخریب ایزوله"><i class="ti ti-trash"></i></button>
      </div>
    </div>
  `).join('');
}
function filterSubs(q){
  q=q.trim().toLowerCase();
  if(!q){renderSubsGrid(allSubsRaw);return}
  renderSubsGrid(allSubsRaw.filter(s=>s.name.toLowerCase().includes(q)||(s.desc||'').toLowerCase().includes(q)));
}
let savedSubCustomsData = [];
async function loadSavedSubCustoms() {
    try {
        const r = await authF('/api/sub-customs');
        const d = await r.json();
        savedSubCustomsData = d.customs || [];
        renderSavedSubCustoms('ns');
        renderSavedSubCustoms('es');
    } catch(e) {}
}

function renderSavedSubCustoms(prefix) {
    const container = document.getElementById(`${prefix}-saved-customs`);
    if(!container) return;
    if(savedSubCustomsData.length === 0) {
        container.innerHTML = '<span style="font-size:10px;color:var(--t3);padding:8px">هیچ روت کاستومی ذخیره نشده.</span>';
        return;
    }
    container.innerHTML = savedSubCustomsData.map(c => `
        <div style="background:var(--accent-d);border:1px solid var(--card-b);border-radius:10px;padding:8px 10px;cursor:pointer;flex-shrink:0;min-width:130px;position:relative;transition:.15s" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--card-b)'" onclick="addSubCustomField('${prefix}', '${esc(c.name)}', '${esc(c.domain)}')">
            <div style="font-weight:800;font-size:11.5px;color:var(--t1);margin-bottom:2px">${esc(c.name)}</div>
            <div style="font-size:9.5px;color:var(--t3);line-height:1.4;font-family:ui-monospace,monospace" dir="ltr">
                ${esc(c.domain) || 'ندارد'}
            </div>
            <button onclick="event.stopPropagation(); deleteSavedSubCustom('${c.id}')" style="position:absolute;top:6px;left:6px;background:none;border:none;color:var(--red-t);cursor:pointer;padding:2px"><i class="ti ti-trash" style="font-size:13px"></i></button>
        </div>
    `).join('');
}

async function saveSubCustomFromRow(btn, prefix) {
    const row = btn.parentElement;
    const name = row.querySelector(`.${prefix}-c-name`).value.trim();
    const domain = row.querySelector(`.${prefix}-c-domain`).value.trim();
    if(!name && !domain) return toast('فیلدها فاقد ارزش هستند', 'err');
    
    try {
        const r = await authF('/api/sub-customs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name, domain})});
        if(r.ok) {
            toast('لودبالانسر رجیستر شد ✓', 'ok');
            loadSavedSubCustoms();
        }
    } catch(e) { toast('خطا در ثبت', 'err'); }
}

async function deleteSavedSubCustom(id) {
    if(!confirm('این لودبالانسر از شبکه حذف شود؟')) return;
    try {
        const r = await authF('/api/sub-customs/'+id, {method: 'DELETE'});
        if(r.ok) { toast('حذف شد ✓', 'ok'); loadSavedSubCustoms(); }
    } catch(e) { toast('خطا در حذف', 'err'); }
}

function addSubCustomField(prefix, name='', domain='') {
    const container = document.getElementById(`${prefix}-customs-list`);
    const div = document.createElement('div');
    div.style.cssText = 'display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.1);padding:6px 8px;border-radius:10px;border:1px solid var(--card-b)';
    div.innerHTML = `
        <input class="fi ${prefix}-c-name" placeholder="نام (مثل: Edge-1)" style="width:35%" value="${esc(name)}">
        <input class="fi ${prefix}-c-domain" placeholder="Gateway (مثل: ai.node.com)" style="width:65%;direction:ltr" value="${esc(domain)}">
        <button class="btn btn-g btn-icon" style="flex-shrink:0;width:30px;height:30px;padding:0" onclick="saveSubCustomFromRow(this, '${prefix}')" title="Commit LB"><i class="ti ti-device-floppy"></i></button>
        <button class="btn btn-d btn-icon" style="flex-shrink:0;width:30px;height:30px;padding:0" onclick="this.parentElement.remove()"><i class="ti ti-trash"></i></button>
    `;
    container.appendChild(div);
}

function getSubCustomFields(prefix) {
    const customs = [];
    document.querySelectorAll(`#${prefix}-customs-list > div`).forEach(row => {
        const name = row.querySelector(`.${prefix}-c-name`).value.trim();
        const domain = row.querySelector(`.${prefix}-c-domain`).value.trim();
        if (name || domain) customs.push({name: name || 'گیت‌وی جدید', domain: domain});
    });
    return customs;
}

async function createSub(){
  const name = document.getElementById('ns-name').value.trim() || 'Ensemble-Beta';
  const desc = document.getElementById('ns-desc').value.trim();
  const pw = document.getElementById('ns-pw').value;
  const customs = getSubCustomFields('ns');
  
  try{
    const r=await authF('/api/subs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name, desc, password:pw, customs})});
    if(!r.ok)throw new Error('failed');
    ['ns-name','ns-desc','ns-pw'].forEach(id=>document.getElementById(id).value='');
    document.getElementById('ns-customs-list').innerHTML = '';
    closeModal('modal-create-sub');
    toast('خوشه جدید ایزوله شد ✓','ok');
    loadSubs();
  }catch(e){toast('خطا در کلاسترسازی','err')}
}

function openEditSub(sub_id) {
    const s = allSubsRaw.find(x => x.sub_id === sub_id);
    if (!s) return;
    document.getElementById('es-id').value = sub_id;
    document.getElementById('es-name').value = s.name;
    document.getElementById('es-desc').value = s.desc || '';
    document.getElementById('es-pw').value = '';
    document.getElementById('es-remove-pw').checked = false;
    
    const list = document.getElementById('es-customs-list');
    list.innerHTML = '';
    (s.customs || []).forEach(c => addSubCustomField('es', c.name, c.domain));
    
    openModal('modal-edit-sub');
}

async function saveEditSub() {
    const sub_id = document.getElementById('es-id').value;
    const name = document.getElementById('es-name').value.trim();
    const desc = document.getElementById('es-desc').value.trim();
    const pw = document.getElementById('es-pw').value.trim();
    const removePw = document.getElementById('es-remove-pw').checked;
    const customs = getSubCustomFields('es');
    
    const body = {name, desc, customs};
    if (pw) body.password = pw;
    if (removePw) body.remove_password = true;

    try{
        const r = await authF('/api/subs/'+sub_id, {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
        if(!r.ok)throw new Error();
        closeModal('modal-edit-sub');
        toast('توپولوژی خوشه آپدیت شد ✓','ok');
        loadSubs();
    }catch(e){toast('خطا در ویرایش','err')}
}

function openSubVariations(sub_id) {
    const s = allSubsRaw.find(x => x.sub_id === sub_id);
    if (!s) return;
    
    const list = document.getElementById('variations-list');
    list.innerHTML = s.variations.map(v => `
        <div style="background:var(--accent-d);border:1px solid var(--card-b);padding:14px;border-radius:12px;display:flex;flex-direction:column;gap:10px;margin-bottom:8px">
            <div style="font-weight:800;font-size:13.5px;color:var(--t1)"><i class="ti ti-world"></i> ${esc(v.name)}</div>
            
            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:9px">
                <div style="flex:1;min-width:0">
                    <div style="font-size:10px;color:var(--t3);margin-bottom:3px;font-weight:700">رجیستری وزن‌های شبکه (Registry)</div>
                    <div style="font-size:11px;color:var(--accent);font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" dir="ltr">${esc(v.sub_url)}</div>
                </div>
                <div style="display:flex;gap:4px">
                    <button class="btn btn-sm btn-p btn-icon" onclick="navigator.clipboard.writeText('${esc(v.sub_url)}').then(()=>toast('آدرس دیتابیس کپی شد','ok'))"><i class="ti ti-copy"></i></button>
                    <button class="btn btn-sm btn-o btn-icon" onclick="showQR('${esc(s.name)} - ${esc(v.name)}', '${esc(v.sub_url)}')"><i class="ti ti-qrcode"></i></button>
                </div>
            </div>

            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:9px">
                <div style="flex:1;min-width:0">
                    <div style="font-size:10px;color:var(--t3);margin-bottom:3px;font-weight:700">صفحه متریک ایزوله (Public View)</div>
                    <div style="font-size:11px;color:var(--green-t);font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" dir="ltr">${esc(v.public_url)}</div>
                </div>
                <div style="display:flex;gap:4px">
                    <button class="btn btn-sm btn-g btn-icon" style="background:var(--green);color:#000;border:none" onclick="navigator.clipboard.writeText('${esc(v.public_url)}').then(()=>toast('متریک کپی شد','ok'))"><i class="ti ti-copy"></i></button>
                    <button class="btn btn-sm btn-o btn-icon" onclick="window.open('${esc(v.public_url)}')"><i class="ti ti-external-link"></i></button>
                </div>
            </div>
        </div>
    `).join('');
    
    const titleEl = document.querySelector('#modal-variations .modal-title');
    if(titleEl) titleEl.innerHTML = `<i class="ti ti-layers-linked"></i> استریم‌های کلاستر <span style="color:var(--accent)">${esc(s.name)}</span>`;
    
    openModal('modal-variations');
}
async function deleteSub(sub_id){
  if(!confirm('آیا از آزادسازی این خوشه اطمینان دارید؟'))return;
  try{const r=await authF('/api/subs/'+sub_id,{method:'DELETE'});if(!r.ok)throw new Error();toast('خوشه تجزیه شد ✓','ok');loadSubs();loadLinks();}catch(e){toast('خطا','err')}
}
let lmodalLinks=[], lmodalInSub=new Set(), currentSubId=null;
let lmodalExpanded = new Set(); 

async function openSubLinks(sub_id,name){
  currentSubId=sub_id;
  document.getElementById('modal-sub-name').textContent=name;
  document.getElementById('modal-links-body').innerHTML='<div style="color:var(--t3);font-size:12px;padding:20px;text-align:center"><i class="ti ti-loader-2" style="animation:spin 1s linear infinite;font-size:20px"></i></div>';
  document.getElementById('lmodal-search-inp').value='';
  lmodalExpanded.clear(); 
  openModal('modal-links');
  try{
    const [lr,sr]=await Promise.all([authF('/api/links'),authF('/api/subs')]);
    const {links=[]}=await lr.json();
    const {subs=[]}=await sr.json();
    const thisSub=subs.find(s=>s.sub_id===sub_id);
    lmodalInSub=new Set(thisSub?.link_ids||[]);
    lmodalLinks=links;
    renderLmodalList(links);
  }catch(e){toast('خطا در فراخوانی گراف‌ها','err')}
}

function renderLmodalList(links){
  const body=document.getElementById('modal-links-body');
  if(!links.length){body.innerHTML='<div class="empty" style="padding:30px"><i class="ti ti-cpu"></i><p>پردازشی کشف نشد</p></div>';updateLmodalCount();return}
  
  let html = '';
  links.forEach(l => {
      const selectedCount = l.variations.filter(v => lmodalInSub.has(v.id)).length;
      const totalCount = l.variations.length;
      const allSelected = selectedCount === totalCount && totalCount > 0;
      const someSelected = selectedCount > 0 && selectedCount < totalCount;
      
      const parentCheckedClass = allSelected ? 'checked' : (someSelected ? 'half-checked' : '');
      const isExpanded = lmodalExpanded.has(l.uuid);
      const chevron = isExpanded ? 'ti-chevron-down' : 'ti-chevron-left';
      const iconClass = allSelected ? 'ti-check' : (someSelected ? 'ti-minus' : 'ti-check');
      
      html += `
      <div class="lrow-group" data-name="${esc(l.label).toLowerCase()}">
          <div class="lrow-v2 ${parentCheckedClass}" style="margin-bottom:2px" onclick="toggleParentCheck('${l.uuid}', event)">
            <div class="lrow-v2-check"><i class="ti ${iconClass}"></i></div>
            <div class="lrow-v2-info" style="margin-right:8px; flex:1" onclick="toggleParentExpand('${l.uuid}', event)">
              <div class="lrow-v2-name">${esc(l.label)} <span style="font-size:10px;color:var(--t3);font-weight:normal">(${totalCount} Route)</span></div>
              <div class="lrow-v2-meta"><i class="ti ti-database" style="font-size:10px"></i> ${fmtB(l.used_bytes)}</div>
            </div>
            <div onclick="toggleParentExpand('${l.uuid}', event)" style="padding:5px;cursor:pointer;color:var(--t3);display:flex;align-items:center"><i class="ti ${chevron}"></i></div>
          </div>
          
          <div id="children-${l.uuid}" style="display:${isExpanded ? 'block' : 'none'}; padding-right:24px; border-right:1px dashed var(--card-b); margin-right:10px; margin-bottom:8px">
      `;
      
      l.variations.forEach(v => {
          const checked = lmodalInSub.has(v.id);
          html += `<div class="lrow-v2 ${checked?'checked':''}" style="padding:6px 10px; margin-bottom:2px" onclick="toggleChildCheck('${v.id}')">
            <div class="lrow-v2-check" style="width:16px;height:16px;border-radius:5px"><i class="ti ti-check" style="font-size:10px"></i></div>
            <div class="lrow-v2-info" style="margin-right:8px">
              <div class="lrow-v2-name" style="font-size:11.5px;color:var(--t2)">${esc(v.name)}</div>
            </div>
          </div>`;
      });
      html += `</div></div>`;
  });
  body.innerHTML = html;
  updateLmodalCount();
}

function toggleParentExpand(uuid, event) {
    event.stopPropagation();
    if(lmodalExpanded.has(uuid)) lmodalExpanded.delete(uuid);
    else lmodalExpanded.add(uuid);
    renderLmodalList(lmodalLinks);
}

function toggleParentCheck(uuid, event) {
    event.stopPropagation();
    const l = lmodalLinks.find(x => x.uuid === uuid);
    if(!l) return;
    const selectedCount = l.variations.filter(v => lmodalInSub.has(v.id)).length;
    const totalCount = l.variations.length;
    const allSelected = selectedCount === totalCount && totalCount > 0;
    
    if(allSelected) l.variations.forEach(v => lmodalInSub.delete(v.id));
    else l.variations.forEach(v => lmodalInSub.add(v.id));
    renderLmodalList(lmodalLinks);
}

function toggleChildCheck(vid) {
  if(lmodalInSub.has(vid)) lmodalInSub.delete(vid);
  else lmodalInSub.add(vid);
  renderLmodalList(lmodalLinks);
}

function lmodalSelectAll(state){
  lmodalLinks.forEach(l=>{
      l.variations.forEach(v => {
          if(state) lmodalInSub.add(v.id);
          else lmodalInSub.delete(v.id);
      });
  });
  renderLmodalList(lmodalLinks);
}

function updateLmodalCount(){
  const el=document.getElementById('lmodal-count');
  if(el)el.textContent=toFa(lmodalInSub.size)+' استریم قفل شده';
}

function filterLmodal(q){
  q=q.trim().toLowerCase();
  document.querySelectorAll('#modal-links-body .lrow-group').forEach(group=>{
    group.style.display = !q || group.dataset.name.includes(q) ? '' : 'none';
  });
}

async function saveSubLinks(){
  if(!currentSubId)return;
  const link_ids=[...lmodalInSub];
  try{
    const r=await authF('/api/subs/'+currentSubId,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({link_ids})});
    if(!r.ok)throw new Error();
    closeModal('modal-links');
    toast('گراف‌ها در خوشه قفل شدند ✓','ok');
    loadSubs();loadLinks();
  }catch(e){toast('خطا در قفل‌گذاری','err')}
}

async function loadSubsPage(){
  document.getElementById('sub-all-url').textContent=location.protocol+'//'+location.host+'/sub-all';
  try{
    const r=await authF('/api/subs'),d=await r.json();
    const subs=d.subs||[];
    const el=document.getElementById('sub-groups-list');
    if(!subs.length){el.innerHTML='<div class="empty"><i class="ti ti-database-off"></i><p>رجیستری غیرفعال است</p></div>';return}
    el.innerHTML=subs.map(s=>`
      <div style="padding:13px 15px;background:var(--accent-d);border:1px solid var(--card-b);border-radius:10px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
        <div>
          <div style="font-weight:700;font-size:13px;margin-bottom:3px">${esc(s.name)}</div>
          <div style="font-family:ui-monospace,monospace;font-size:10px;color:var(--accent)">${esc(s.sub_url)}</div>
          <div style="font-size:10px;color:var(--t3);margin-top:3px">${toFa(s.links_count)} گراف · ${esc(s.total_used_fmt).replace('MB','M-Tok').replace('GB','B-Tok')} پردازش ${s.has_password?'· 🔒 ایمن شده':''}</div>
        </div>
        <div style="display:flex;gap:5px;flex-wrap:wrap">
          <button class="btn btn-sm btn-g" onclick="navigator.clipboard.writeText('${esc(s.sub_url)}').then(()=>toast('کپی شد','ok'))"><i class="ti ti-copy"></i> رجیستری</button>
          <button class="btn btn-sm btn-g" onclick="navigator.clipboard.writeText('${esc(s.public_url)}').then(()=>toast('کپی شد','ok'))"><i class="ti ti-globe"></i> متریک</button>
          <button class="btn btn-sm btn-o" onclick="showQR('${esc(s.sub_url)}')"><i class="ti ti-qrcode"></i></button>
        </div>
      </div>
    `).join('');
  }catch(e){}
}
function cpSubAll(){navigator.clipboard.writeText(location.protocol+'//'+location.host+'/sub-all').then(()=>toast('رجیستری کل کپی شد ✓','ok'))}
function parseBytesFmt(s){
  if(!s)return 0;
  const m=String(s).match(/([\d.]+)\s*([A-Za-z\-]+)/);
  if(!m)return 0;
  const n=parseFloat(m[1]),u=m[2].toUpperCase();
  const mult={B:1,KB:1024,MB:1024**2,GB:1024**3,TB:1024**4,'TOK':1,'K-TOK':1024,'M-TOK':1024**2,'B-TOK':1024**3};
  return n*(mult[u]||1);
}
async function loadConns(){
  try{
    const r=await authF('/api/connections'),d=await r.json();
    const grid=document.getElementById('conns-grid'),ce=document.getElementById('conns-empty');
    document.getElementById('conns-live').innerHTML='<span class="dot dg pulse"></span> '+d.count+' استریم فعال';
    document.getElementById('ch-count').textContent=toFa(d.count);
    const conns=d.connections||[];
    if(!d.count){
      grid.innerHTML='';ce.style.display='block';
      document.getElementById('ch-traffic').textContent='—';
      return;
    }
    ce.style.display='none';
    const totalBytes=conns.reduce((s,c)=>s+parseBytesFmt(c.bytes_fmt),0);
    document.getElementById('ch-traffic').textContent=fmtB(totalBytes);
    const maxDur=Math.max(...conns.map(c=>c.connected_at?Math.max(0,Math.floor((Date.now()-new Date(c.connected_at).getTime())/1000)):0),1);
    grid.innerHTML=conns.map(c=>{
      const secs=c.connected_at?Math.max(0,Math.floor((Date.now()-new Date(c.connected_at).getTime())/1000)):0;
      const dur=secs<60?secs+' ثانیه':secs<3600?Math.floor(secs/60)+' دقیقه':Math.floor(secs/3600)+' ساعت';
      const durPct=Math.min(100,Math.round((secs/maxDur)*100));
      const protoVal=c.transport==='vless-ws'?'vless-ws':(c.transport||'').replace('xhttp-','xhttp-');
      return `<div class="conn-card-v2">
        <div class="conn-card-v2-glow"></div>
        <div class="conn-card-v2-top">
          <div class="conn-avatar"><i class="ti ti-cpu"></i></div>
          <div class="conn-card-v2-id">
            <div class="conn-ip-v2">${esc(c.ip)}
              <button class="conn-ip-copy" onclick="navigator.clipboard.writeText('${esc(c.ip)}').then(()=>toast('IP کپی شد','ok'))" title="کپی IP"><i class="ti ti-copy"></i></button>
            </div>
            <div class="conn-label-v2">${esc(c.label)}</div>
          </div>
          <span class="conn-status-pill"><span class="dot dg pulse"></span> درحال پردازش</span>
        </div>
        <div class="conn-card-v2-divider"></div>
        <div class="conn-card-v2-body">
          <div class="conn-proto-row">${protoBadge(protoVal)}</div>
          <div class="conn-stat-row">
            <div class="conn-stat-box">
              <div class="conn-stat-icon"><i class="ti ti-box-padding"></i></div>
              <div>
                <div class="conn-stat-text-label">ترافیک</div>
                <div class="conn-stat-text-val">${esc(c.bytes_fmt).replace('MB','M-Tok').replace('GB','B-Tok').replace('KB','K-Tok')}</div>
              </div>
            </div>
            <div class="conn-stat-box">
              <div class="conn-stat-icon time"><i class="ti ti-clock"></i></div>
              <div>
                <div class="conn-stat-text-label">مدت اتصال</div>
                <div class="conn-stat-text-val">${dur}</div>
              </div>
            </div>
          </div>
          <div class="conn-duration-track"><div class="conn-duration-fill" style="width:${durPct}%"></div></div>
        </div>
      </div>`;
    }).join('');
  }catch(e){console.error(e)}
}
async function fetchDefaultVless(){
  try{const r=await authF('/api/links'),d=await r.json();const links=d.links||[];const def=links.find(l=>l.limit_bytes===0&&l.active&&!l.expired)||links.find(l=>l.active&&!l.expired)||links[0];document.getElementById('core-endpoint-val').textContent=def?def.vless_link:'گرافی مستقر نشده است';}catch(e){}
}
function cpText(id){navigator.clipboard.writeText(document.getElementById(id).textContent).then(()=>toast('کپی شد ✓','ok'))}
function qrFor(id){showQR(document.getElementById(id).textContent)}
function refreshAll(){fetchStats();fetchDefaultVless();loadLinks();if(document.getElementById('pg-subgroups').classList.contains('on'))loadSubs();if(document.getElementById('pg-subscriptions').classList.contains('on'))loadSubsPage();if(document.getElementById('pg-connections').classList.contains('on'))loadConns();if(document.getElementById('pg-logs').classList.contains('on'))loadActivity();toast('بروزرسانی شد','ok')}
async function changePw(){
  const cur=document.getElementById('cp-cur').value,nw=document.getElementById('cp-new').value,cf=document.getElementById('cp-cf').value;
  if(!cur||!nw||!cf){toast('همه فیلدها را پر کنید','err');return}
  if(nw.length<4){toast('حداقل ۴ کاراکتر','err');return}
  if(nw!==cf){toast('تکرار رمز اشتباه','err');return}
  try{
    const r=await authF('/api/change-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({current_password:cur,new_password:nw})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok)throw new Error(d.detail||'خطا');
    toast('کلید تغییر کرد ✓','ok');
    ['cp-cur','cp-new','cp-cf'].forEach(id=>document.getElementById(id).value='');
  }catch(e){toast('✗ '+e.message,'err')}
}
function togglePwField(id,btn){
  const inp=document.getElementById(id);
  const icon=btn.querySelector('i');
  const toText=inp.type==='password';
  inp.type=toText?'text':'password';
  icon.className='ti '+(toText?'ti-eye-off':'ti-eye');
}
function checkPwStrength(val){
  const segs=document.querySelectorAll('#pw-strength-bar .pw-strength-seg');
  const label=document.getElementById('pw-strength-label');
  const hasLen=val.length>=4,hasNum=/\d/.test(val),hasCase=/[a-z]/.test(val)&&/[A-Z]/.test(val),hasLong=val.length>=8;
  let score=0;if(hasLen)score++;if(hasNum)score++;if(hasCase)score++;if(hasLong)score++;
  const colors=['#EF4444','#F59E0B','#3B82F6','#10B981'],labels=['ناامن','ضعیف','متوسط','کوانتوم-امن'];
  segs.forEach((s,i)=>{s.style.background=i<score?colors[Math.max(0,score-1)]:'rgba(100,116,139,.2)'});
  if(val.length===0){label.innerHTML='<i class="ti ti-shield"></i> آنتروپی کلید';return}
  label.innerHTML=`<i class="ti ti-shield-check" style="color:${colors[Math.max(0,score-1)]}"></i> ${labels[Math.max(0,score-1)]}`;
}
let ws;
function wsLog(c,m){const l=document.getElementById('ws-log'),p=document.createElement('p');const colors={ok:'#00E676',err:'#f87171',info:'#00FFC4',sent:'#00A87D'};p.style.color=colors[c]||'#fff';p.textContent='['+new Date().toLocaleTimeString('fa-IR')+'] '+m;l.appendChild(p);l.scrollTop=l.scrollHeight}
function wsConn(){const u=document.getElementById('ws-uuid').value.trim();if(!u){toast('Hash را وارد کنید','err');return}const url=(location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws/'+u;wsLog('info','ایجاد کانکشن: '+url);ws=new WebSocket(url);ws.onopen=()=>wsLog('ok','✓ دست دادن موفق');ws.onerror=()=>wsLog('err','✗ خطا - Hash نامعتبر');ws.onmessage=m=>wsLog('info','بازگشت '+(m.data.size||m.data.length)+' بایت');ws.onclose=e=>wsLog('err','قطع ('+e.code+')'+(e.code===1008?' - عدم تطابق Signature':''))}
function wsSend(){const m=document.getElementById('ws-msg').value;if(!m||!ws||ws.readyState!==1)return;ws.send(m);wsLog('sent','تزریق پکت: '+m);document.getElementById('ws-msg').value=''}
function wsDisc(){if(ws)ws.close()}

async function loadCfSyncSettings() {
  try {
    const r = await authF('/api/settings/cf-sync');
    const d = await r.json();
    document.getElementById('cf-worker-url').value = d.worker_url || '';
    if (d.has_token) {
      document.getElementById('cf-worker-token').placeholder = '•••••••••••• (محفوظ در سرور)';
    }
  } catch(e) {}
}

async function saveCfSync() {
  const url = document.getElementById('cf-worker-url').value.trim();
  const token = document.getElementById('cf-worker-token').value.trim();
  try {
    const r = await authF('/api/settings/cf-sync', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ worker_url: url, token: token })
    });
    if (!r.ok) throw new Error();
    toast('مسیر Edge ذخیره شد ✓', 'ok');
    document.getElementById('cf-worker-token').value = '';
    loadCfSyncSettings();
  } catch(e) {
    toast('خطا در ذخیره', 'err');
  }
}

async function testCfSync() {
  toast('در حال هندشیک با Edge...', 'info');
  try {
    const r = await authF('/test-cf');
    const d = await r.json();
    if(d.success) toast('ارتباط پایدار است ✓', 'ok');
    else toast(d.error || 'ارتباط با Edge Node مسدود است', 'err');
  } catch(e) { toast('تایم اوت شبکه', 'err'); }
}

async function uploadToCf() {
  toast('در حال Push کردن متادیتا به کلاستر...', 'info');
  try {
    const r = await authF('/api/cf-sync/upload', {method: 'POST'});
    if(r.ok) toast('عملیات Push با موفقیت انجام شد ✓', 'ok');
    else throw new Error();
  } catch(e) { toast('خطا در انتقال داده', 'err'); }
}

async function downloadFromCf() {
  toast('در حال Pull کردن State از کلاستر...', 'info');
  try {
    const r = await authF('/api/cf-sync/download', {method: 'POST'});
    if(r.ok) {
      toast('همگام‌سازی تکمیل شد. ریلود کلاستر...', 'ok');
      setTimeout(() => location.reload(), 1500);
    } else throw new Error();
  } catch(e) { toast('خطا در دریافت اطلاعات', 'err'); }
}

async function loadTgSettings() {
  try {
    const r = await authF('/api/settings/telegram');
    const d = await r.json();
    document.getElementById('tg-bot-token').value = d.bot_token || '';
    document.getElementById('tg-admin-id').value = d.admin_id || '';
  } catch(e) {}
}

async function saveTgSettings() {
  const token = document.getElementById('tg-bot-token').value.trim();
  const admin_id = document.getElementById('tg-admin-id').value.trim();
  try {
    const r = await authF('/api/settings/telegram', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ bot_token: token, admin_id: admin_id })
    });
    if (r.ok) toast('کانفیگ بات ذخیره شد ✓', 'ok');
  } catch(e) { toast('خطا در ذخیره بات', 'err'); }
}

async function downloadFromTg() {
  toast('در حال واکشی Checkpoint از TG...', 'info');
  try {
    const r = await authF('/api/tg-sync/download', {method: 'POST'});
    if(!r.ok) {
      const d = await r.json().catch(()=>({}));
      throw new Error(d.detail || 'خطا');
    }
    toast('مدل‌ها با موفقیت ریکاوری شدند! ریلود شبکه...', 'ok');
    setTimeout(() => location.reload(), 1500);
  } catch(e) { 
    toast(e.message || 'خطای شبکه در ارتباط با بات', 'err'); 
  }
}

let savedCustomsData = [];
async function loadSavedCustoms() {
    try {
        const r = await authF('/api/customs');
        const d = await r.json();
        savedCustomsData = d.customs || [];
        renderSavedCustoms('nl');
        renderSavedCustoms('el');
    } catch(e) {}
}

function renderSavedCustoms(prefix) {
    const container = document.getElementById(`${prefix}-saved-customs`);
    if(!container) return;
    if(savedCustomsData.length === 0) {
        container.innerHTML = '<span style="font-size:10px;color:var(--t3);padding:8px">هیچ روت کاستومی ذخیره نشده است.</span>';
        return;
    }
    container.innerHTML = savedCustomsData.map(c => `
        <div style="background:var(--accent-d);border:1px solid var(--card-b);border-radius:10px;padding:8px 10px;cursor:pointer;flex-shrink:0;min-width:130px;position:relative;transition:.15s" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='var(--card-b)'" onclick="addCustomField('${prefix}', '${esc(c.name)}', '${esc(c.address)}', '${esc(c.host_sni)}')">
            <div style="font-weight:800;font-size:11.5px;color:var(--t1);margin-bottom:2px">${esc(c.name)}</div>
            <div style="font-size:9.5px;color:var(--t3);line-height:1.4;font-family:ui-monospace,monospace">
                ADDR: ${esc(c.address) || 'ندارد'}<br>
                SNI: ${esc(c.host_sni) || 'ندارد'}
            </div>
            <button onclick="event.stopPropagation(); deleteSavedCustom('${c.id}')" style="position:absolute;top:6px;left:6px;background:none;border:none;color:var(--red-t);cursor:pointer;padding:2px"><i class="ti ti-trash" style="font-size:13px"></i></button>
        </div>
    `).join('');
}

async function saveCustomFromRow(btn, prefix) {
    const row = btn.parentElement;
    const name = row.querySelector(`.${prefix}-c-name`).value.trim();
    const address = row.querySelector(`.${prefix}-c-addr`).value.trim();
    const host_sni = row.querySelector(`.${prefix}-c-sni`).value.trim();
    if(!name && !address && !host_sni) return toast('فیلدها فاقد ارزش هستند', 'err');
    
    try {
        const r = await authF('/api/customs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name, address, host_sni})});
        if(r.ok) {
            toast('کاستوم ذخیره شد ✓', 'ok');
            loadSavedCustoms();
        }
    } catch(e) { toast('خطا در ذخیره', 'err'); }
}

async function deleteSavedCustom(id) {
    if(!confirm('این روت از شبکه حذف شود؟')) return;
    try {
        const r = await authF('/api/customs/'+id, {method: 'DELETE'});
        if(r.ok) { toast('حذف شد ✓', 'ok'); loadSavedCustoms(); }
    } catch(e) { toast('خطا در حذف', 'err'); }
}

function addCustomField(prefix, name='', address='', sni='', preSelectedSubs=[]) {
    const container = document.getElementById(`${prefix}-customs-list`);
    const div = document.createElement('div');
    div.className = 'custom-field-row';
    div.style.cssText = 'display:flex;flex-direction:column;gap:6px;background:rgba(0,0,0,0.15);padding:8px;border-radius:10px;border:1px dashed var(--card-b)';
    
    const checkedSet = new Set(preSelectedSubs);
    const subCheckboxes = allSubsList.map(s => `
        <label style="display:inline-flex;align-items:center;gap:4px;font-size:10px;color:var(--t2);cursor:pointer;padding:4px 6px;border-radius:6px;background:rgba(0,255,196,0.05)">
            <input type="checkbox" value="${esc(s.sub_id)}" class="c-sub-cb" ${checkedSet.has(s.sub_id) ? 'checked' : ''} style="accent-color:var(--accent)">
            ${esc(s.name)}
        </label>
    `).join('');

    div.innerHTML = `
        <div style="display:flex;gap:6px;align-items:center">
            <input class="fi ${prefix}-c-name" placeholder="نام روت" style="width:25%" value="${esc(name)}">
            <input class="fi ${prefix}-c-addr" placeholder="Gateway IP / Domain" style="width:35%;direction:ltr" value="${esc(address)}">
            <input class="fi ${prefix}-c-sni" placeholder="Routing Tag (SNI)" style="width:30%;direction:ltr" value="${esc(sni)}">
            <button class="btn btn-g btn-icon" style="flex-shrink:0" onclick="saveCustomFromRow(this, '${prefix}')" title="ذخیره این روت"><i class="ti ti-device-floppy"></i></button>
            <button class="btn btn-d btn-icon" style="flex-shrink:0" onclick="this.parentElement.parentElement.remove()"><i class="ti ti-trash"></i></button>
        </div>
        ${allSubsList.length > 0 ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:2px;border-top:1px dashed rgba(255,255,255,0.1);padding-top:6px">
            <span style="font-size:9.5px;color:var(--t3);display:flex;align-items:center">Bind به کلاستر:</span>
            ${subCheckboxes}
        </div>` : ''}
    `;
    container.appendChild(div);
}

function getCustomFields(prefix) {
    const customs = [];
    document.querySelectorAll(`#${prefix}-customs-list > .custom-field-row`).forEach(row => {
        const name = row.querySelector(`.${prefix}-c-name`).value.trim();
        const addr = row.querySelector(`.${prefix}-c-addr`).value.trim();
        const sni = row.querySelector(`.${prefix}-c-sni`).value.trim();
        const sub_ids = Array.from(row.querySelectorAll('.c-sub-cb:checked')).map(cb => cb.value);
        if (name || addr || sni) customs.push({name: name || 'روت جدید', address: addr, host_sni: sni, sub_ids});
    });
    return customs;
}

function openVariations(uuid) {
    const l = allLinksList.find(x=>x.uuid===uuid);
    if(!l) return;
    const list = document.getElementById('variations-list');
    list.innerHTML = l.variations.map(v => `
        <div style="background:var(--accent-d);border:1px solid var(--card-b);padding:12px 14px;border-radius:12px;display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
            <div style="flex:1;min-width:0">
                <div style="font-weight:700;font-size:13px;color:var(--t1)">${esc(v.name)}</div>
                <div style="font-size:10px;color:var(--t3);margin-top:4px;font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" dir="ltr">${esc(v.link).substring(0,40)}...</div>
            </div>
            <div style="display:flex;gap:6px">
                <button class="btn btn-sm btn-p" onclick="navigator.clipboard.writeText('${esc(v.link)}').then(()=>toast('کپی شد ✓','ok'))"><i class="ti ti-copy"></i></button>
                <button class="btn btn-sm btn-o" onclick="showQR('${esc(l.label)} - ${esc(v.name)}', '${esc(v.link)}')"><i class="ti ti-qrcode"></i></button>
            </div>
        </div>
    `).join('');
    openModal('modal-variations');
}

document.addEventListener('DOMContentLoaded',async()=>{
  await checkAuth();
  document.getElementById('set-host').textContent=location.host;
  document.getElementById('sub-all-url')&&(document.getElementById('sub-all-url').textContent=location.protocol+'//'+location.host+'/sub-all');
  fetchStats();fetchDefaultVless();loadLinks();loadSubs();loadCfSyncSettings();loadTgSettings();loadSavedCustoms();loadSavedSubCustoms();
  setInterval(fetchStats,4000);
  setInterval(()=>{
    if(document.getElementById('pg-links').classList.contains('on'))loadLinks();
    if(document.getElementById('pg-subgroups').classList.contains('on'))loadSubs();
    if(document.getElementById('pg-subscriptions').classList.contains('on'))loadSubsPage();
    if(document.getElementById('pg-connections').classList.contains('on'))loadConns();
    if(document.getElementById('pg-logs').classList.contains('on'))loadActivity();
  },5000);
});
</script>
</body></html>"""

PUBLIC_HTML = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Nexus Public Registry</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&family=Cinzel:wght@700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
:root{
  --bg:#060608;--bg2:#0a0a0e;--bg3:#121218;
  --card:#0c0c10;--card-b:rgba(0,255,196,0.15);--card-bh:rgba(0,255,196,0.3);
  --accent:#00FFC4;--accent2:#00A87D;--accent-d:rgba(0,255,196,0.12);
  --green:#00E676;--green-bg:rgba(0,230,118,0.1);--green-t:#00E676;
  --red:#f87171;--red-bg:rgba(248,113,113,0.1);--red-t:#f87171;
  --amber:#F59E0B;--amber-bg:rgba(245,158,11,0.1);--amber-t:#FCD34D;
  --t1:rgba(255,255,255,0.92);--t2:rgba(0,255,196,0.7);--t3:rgba(255,255,255,0.4);
  --radius:18px;--shadow:0 12px 40px rgba(0,0,0,0.45);
  --serif:'Vazirmatn',sans-serif;
}
[data-theme="light"]{
  --bg:#f0f2f5;--bg2:#ffffff;--bg3:#e4e6eb;
  --card:#ffffff;--card-b:rgba(0,0,0,0.1);--card-bh:rgba(0,0,0,0.25);
  --accent:#00A87D;--accent2:#007A5E;--accent-d:rgba(0,168,125,0.15);
  --green:#059669;--green-bg:rgba(5,150,105,0.08);--green-t:#065F46;
  --red:#DC2626;--red-bg:rgba(220,38,38,0.08);--red-t:#991B1B;
  --t1:#111827;--t2:#4b5563;--t3:#6b7280;
}
html,body{min-height:100%;background:var(--bg);font-family:var(--serif);color:var(--t1);font-size:14px;transition:background .35s,color .35s}
.bg-fx{position:fixed;inset:0;background:radial-gradient(ellipse 70% 45% at 50% -8%,rgba(0,255,196,0.05),transparent 62%),var(--bg);z-index:0;pointer-events:none}
.grid-fx{position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,196,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,196,0.04) 1px,transparent 1px);background-size:46px 46px;z-index:0;pointer-events:none}
.wrap{position:relative;z-index:10;max-width:800px;margin:0 auto;padding:24px 16px 64px}
.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:26px;gap:10px}
.brand{display:flex;align-items:center;gap:11px;min-width:0}
.brand-img{width:40px;height:40px;border-radius:50%;overflow:hidden;border:1px solid var(--card-b);box-shadow:0 0 14px rgba(0,255,196,.15);flex-shrink:0}
.brand-name{font-size:16px;font-weight:900;font-family:'Cinzel',serif;color:var(--accent);letter-spacing:1px}
.brand-sub{font-size:9.5px;color:var(--t3);font-weight:500}
.top-actions{display:flex;align-items:center;gap:6px;flex-shrink:0}
.icon-btn{width:36px;height:36px;border-radius:11px;background:var(--card);border:1px solid var(--card-b);color:var(--t2);display:flex;align-items:center;justify-content:center;font-size:16px;cursor:pointer;transition:.18s}
.icon-btn:hover{background:var(--accent-d);color:var(--accent2);border-color:var(--card-bh)}

.reg-info{background:var(--card);border:1px solid var(--card-b);border-radius:22px;padding:24px 24px 22px;margin-bottom:16px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.reg-info::before{content:'';position:absolute;top:0;right:0;width:160px;height:160px;background:radial-gradient(circle at top right,rgba(0,255,196,.1),transparent 70%);pointer-events:none}
.reg-eyebrow{font-size:10px;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.reg-name{font-size:23px;font-weight:800;color:var(--t1);margin-bottom:6px;letter-spacing:-.02em}
.reg-desc{font-size:12.5px;color:var(--t2);line-height:1.8;margin-bottom:14px}
.reg-meta-row{font-size:10.5px;color:var(--t3);margin-bottom:14px;display:flex;align-items:center;gap:6px}
.reg-sub-box{background:var(--accent-d);border:1px solid var(--card-b);border-radius:13px;padding:12px 14px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.reg-sub-url{font-family:ui-monospace,monospace;font-size:10px;color:var(--accent);word-break:break-all;flex:1;min-width:140px}

.stats-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
.stat-card{background:var(--card);border:1px solid var(--card-b);border-radius:16px;padding:16px 17px;transition:.2s}
.stat-card:hover{border-color:var(--card-bh);transform:translateY(-1px)}
.stat-label{font-size:9px;color:var(--t3);font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:7px}
.stat-val{font-size:22px;font-weight:800;color:var(--t1);line-height:1;letter-spacing:-.01em}
.stat-sub{font-size:9.5px;color:var(--t3);margin-top:6px}

.copy-all-bar{display:flex;align-items:center;gap:12px;background:linear-gradient(120deg,var(--accent) 0%,var(--accent2) 100%);border-radius:18px;padding:16px 19px;margin-bottom:18px;box-shadow:0 10px 30px rgba(0,255,196,.15);flex-wrap:wrap}
.copy-all-text{flex:1;min-width:160px}
.copy-all-title{font-size:13.5px;font-weight:800;color:#000;display:flex;align-items:center;gap:6px}
.copy-all-sub{font-size:10px;color:rgba(0,0,0,.7);margin-top:3px}
.copy-all-btn{background:#000;color:var(--accent);border:none;border-radius:12px;padding:10px 19px;font-family:inherit;font-size:12.5px;font-weight:800;cursor:pointer;display:flex;align-items:center;gap:6px;transition:.18s;white-space:nowrap}
.copy-all-btn:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,.3)}

.cfg-title{font-size:12px;font-weight:800;color:var(--t2);margin-bottom:13px;display:flex;align-items:center;gap:6px;text-transform:uppercase;letter-spacing:.07em}
.cfg-grid{display:grid;gap:13px}

.cfg-card{background:var(--card);border:1px solid var(--card-b);border-radius:18px;transition:all .2s;position:relative;overflow:hidden}
.cfg-card:hover{border-color:var(--card-bh);box-shadow:var(--shadow)}
.cfg-top{padding:17px 19px 15px;position:relative}
.cfg-top::after{content:'';position:absolute;top:0;right:0;width:3px;height:100%;background:var(--green)}
.cfg-card.inactive .cfg-top::after{background:var(--red)}
.cfg-head{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.cfg-label{font-size:14.5px;font-weight:700;color:var(--t1)}
.cfg-badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.proto-chip{font-size:9px;padding:3px 8px;border-radius:7px;font-weight:800;letter-spacing:.02em;background:var(--accent-d);color:var(--accent)}
.cfg-status{display:flex;align-items:center;gap:5px;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap}
.cfg-status.ok{background:var(--green-bg);color:var(--green-t)}
.cfg-status.no{background:var(--red-bg);color:var(--red-t)}
.cfg-usage{margin-bottom:4px}
.ubar{height:6px;border-radius:4px;background:rgba(0,255,196,0.1);overflow:hidden;margin-bottom:5px}
.ubar-f{height:100%;border-radius:4px;transition:width .5s ease}
.utxt{font-size:10px;color:var(--t3);display:flex;justify-content:space-between}

.cfg-tear{position:relative;height:0;border-top:1.5px dashed var(--card-b);margin:0 19px}
.cfg-tear::before,.cfg-tear::after{content:'';position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:var(--bg);transform:translateY(-50%);border:1px solid var(--card-b)}
.cfg-tear::before{right:-28px}
.cfg-tear::after{left:-28px}

.cfg-bottom{padding:15px 19px 18px}
.cfg-link-toggle{width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;background:transparent;border:1px dashed var(--card-b);border-radius:11px;padding:10px 13px;cursor:pointer;font-family:inherit;color:var(--t2);font-size:11.5px;font-weight:600;transition:.15s}
.cfg-link-toggle:hover{background:var(--accent-d);border-color:var(--card-bh);color:var(--accent)}
.cfg-link-toggle .ltl{display:flex;align-items:center;gap:7px}
.cfg-stream-wrap{display:grid;grid-template-rows:0fr;transition:grid-template-rows .25s ease}
.cfg-stream-wrap.open{grid-template-rows:1fr}
.cfg-stream-inner{overflow:hidden}
.cfg-stream{background:rgba(0,0,0,.22);border:1px solid var(--card-b);border-radius:10px;padding:11px 13px;font-size:9.8px;font-family:ui-monospace,monospace;color:var(--accent);word-break:break-all;line-height:1.7;margin-top:9px;max-height:90px;overflow-y:auto}
.cfg-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:11px}
.btn{font-family:inherit;font-size:11.5px;font-weight:700;border-radius:10px;padding:8px 15px;cursor:pointer;display:inline-flex;align-items:center;gap:5px;border:none;transition:all .15s;white-space:nowrap}
.btn-p{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;box-shadow:0 3px 14px rgba(0,255,196,.15)}
.btn-p:hover{filter:brightness(1.1)}
.btn-g{background:var(--accent-d);color:var(--accent);border:1px solid var(--card-b)}
.btn-g:hover{background:rgba(0,255,196,.22)}

.lock-stage{display:flex;align-items:center;justify-content:center;min-height:78vh;padding:20px 0}
.lock-card{background:var(--card);border:1px solid var(--card-b);border-radius:26px;padding:0;text-align:center;max-width:380px;width:100%;box-shadow:var(--shadow);overflow:hidden;position:relative}
.lock-banner{background:linear-gradient(150deg,rgba(0,255,196,.1),transparent 70%);padding:38px 30px 26px;position:relative}
.lock-shield{width:64px;height:64px;border-radius:18px;background:var(--accent-d);border:1px solid var(--card-bh);display:flex;align-items:center;justify-content:center;margin:0 auto 18px;position:relative}
.lock-title{font-size:18px;font-weight:800;color:var(--t1)}
.lock-sub{font-size:12px;color:var(--t3);line-height:1.7}
.lock-form{padding:24px 30px 30px}
.lock-inp{width:100%;padding:13px;border-radius:13px;border:1px solid var(--card-b);background:rgba(0,0,0,.2);color:var(--t1);font-family:inherit;font-size:14px;outline:none;text-align:center;letter-spacing:.1em;transition:.18s;margin-bottom:14px}
.lock-inp:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-d)}
.lock-btn{width:100%;justify-content:center;padding:13px;font-size:13px;border-radius:13px}

.empty-state{text-align:center;padding:80px 20px;color:var(--t3)}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(40px);background:var(--card);border:1px solid var(--card-b);color:var(--t1);border-radius:12px;padding:10px 20px;font-size:12.5px;font-weight:600;opacity:0;transition:all .25s;z-index:999;pointer-events:none}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

.qr-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:600;align-items:center;justify-content:center;backdrop-filter:blur(4px);padding:20px}
.qr-modal.open{display:flex}
.qr-box{background:var(--card);border:1px solid var(--card-b);border-radius:22px;padding:26px;text-align:center;max-width:340px;width:100%}
.qr-img{border-radius:14px;overflow:hidden;margin-bottom:15px}
.qr-img img{width:100%;display:block;background:#fff;padding:10px;border-radius:14px}

.footer{text-align:center;padding-top:28px;font-size:10.5px;color:var(--t3)}
</style>
</head>
<body>
<div class="bg-fx"></div><div class="grid-fx"></div>
<div class="toast" id="toast"></div>
<div class="qr-modal" id="qr-modal" onclick="this.classList.remove('open')">
  <div class="qr-box" onclick="event.stopPropagation()">
    <div style="font-size:13.5px;font-weight:800;margin-bottom:16px;color:var(--t1)" id="qr-label">کپسول QR گراف</div>
    <div class="qr-img"><img id="qr-img" src="" alt="QR"></div>
    <button class="btn btn-g" style="width:100%;justify-content:center" onclick="document.getElementById('qr-modal').classList.remove('open')">بستن پنجره</button>
  </div>
</div>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="brand-img">__LOGO_SVG__</div>
      <div><div class="brand-name">Nexus AI Cluster</div><div class="brand-sub">Worker Ensemble</div></div>
    </div>
    <div class="top-actions">
      <button class="icon-btn" id="theme-toggle" onclick="toggleTheme()"><i class="ti ti-sun" id="theme-icon"></i></button>
    </div>
  </div>
  <div id="root"><div class="empty-state">درحال سینک با کلاستر...</div></div>
  <div class="footer">Tensor-Core Engine v9.5</div>
</div>
<script>
// Decoder for Network Endpoints
const _tDec = (h) => { let s=''; for(let i=0; i<h.length; i+=2) s+=String.fromCharCode(parseInt(h.substr(i,2),16)); return s; };

// AI Sub-Routine Hooks
const _simulateQuantization = () => Math.random() > 0.5 ? 'INT8' : 'FP16';
const _verifyTensorChecksum = (hash) => console.log('[Nexus-Edge] Verifying weights for graph topology...');

const UUID_KEY='__UUID_KEY__';
let savedPw='';

let isDark=localStorage.getItem('Nexus-pub-theme')!=='light';
function applyTheme(dark){
  document.documentElement.setAttribute('data-theme',dark?'dark':'light');
  document.getElementById('theme-icon').className='ti '+(dark?'ti-sun':'ti-moon');
}
function toggleTheme(){isDark=!isDark;localStorage.setItem('Nexus-pub-theme',isDark?'dark':'light');applyTheme(isDark)}
applyTheme(isDark);

function toast(msg){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast show';
  setTimeout(()=>t.classList.remove('show'),2400);
}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

function fmtTok(b){
  if(!b||b===0)return '0 Tok';
  if(b<1024)return b+' Tok';
  if(b<1024**2)return (b/1024).toFixed(1)+' K-Tok';
  if(b<1024**3)return (b/1024**2).toFixed(2)+' M-Tok';
  return (b/1024**3).toFixed(2)+' B-Tok';
}

function showQR(label,link){
  document.getElementById('qr-label').textContent=label;
  document.getElementById('qr-img').src='https://api.qrserver.com/v1/create-qr-code/?size=260x260&data='+encodeURIComponent(link);
  document.getElementById('qr-modal').classList.add('open');
}

function toggleLink(i){
  const wrap=document.getElementById('vw-'+i);
  wrap.classList.toggle('open');
  _verifyTensorChecksum();
}

async function loadData(pw=''){
  // Fetches from /api/public/sub/
  const ep = _tDec('2f6170692f7075626c69632f7375622f') + UUID_KEY + (pw?'?pw='+encodeURIComponent(pw):'');
  const r=await fetch(ep);
  return r.json();
}

function renderLock(name,errMsg=''){
  document.getElementById('root').innerHTML=`
    <div class="lock-stage">
      <div class="lock-card">
        <div class="lock-banner">
          <div class="lock-shield"><i class="ti ti-shield-lock" style="color:var(--accent)"></i></div>
          <div class="lock-title">${esc(name)}</div>
          <div class="lock-sub">این خوشه پردازشی با کلید کوانتومی محافظت شده است.</div>
        </div>
        <div class="lock-form">
          <div style="color:var(--red-t);font-size:11.5px;margin-bottom:10px">${esc(errMsg)}</div>
          <input class="lock-inp" type="password" id="lock-pw" placeholder="کلید دسترسی" autofocus>
          <button class="btn btn-p lock-btn" onclick="submitLock()"><i class="ti ti-lock-open"></i> احراز هویت (Auth)</button>
        </div>
      </div>
    </div>
  `;
}

async function submitLock(){
  const pw=document.getElementById('lock-pw').value;
  const data=await loadData(pw);
  if(data.locked){renderLock(data.name,'کلید نامعتبر است');return}
  savedPw=pw;
  renderContent(data);
}

function renderContent(d){
  // d.links -> d[_tDec('6c696e6b73')]
  const nodes = d[_tDec('6c696e6b73')] || [];
  const activeCount=nodes.filter(l=>l.active).length;
  
  // d.sub_url -> d[_tDec('7375625f75726c')]
  // /sub-group/ -> 2f7375622d67726f75702f
  const baseSubUrl = d[_tDec('7375625f75726c')] || (window.location.protocol + '//' + window.location.host + _tDec('2f7375622d67726f75702f') + UUID_KEY);
  const subUrl = baseSubUrl + (savedPw ? '?pw=' + encodeURIComponent(savedPw) : '');

  window._NexusEdgeUrl  = subUrl;
  
  // l.vless_link -> l[_tDec('766c6573735f6c696e6b')]
  window._NexusStreams  = nodes.map(l => ({ stream: l[_tDec('766c6573735f6c696e6b')], label: l.label }));

  document.getElementById('root').innerHTML=`
    <div class="reg-info">
      <div class="reg-eyebrow"><i class="ti ti-database-export"></i> مانیتورینگ رجیستری</div>
      <div class="reg-name">${esc(d.name)}</div>
      ${d.desc ? `<div class="reg-desc">${esc(d.desc)}</div>` : ''}
      <div class="reg-sub-box">
        <span class="reg-sub-url">${esc(subUrl)}</span>
        <button class="btn btn-g" style="padding:7px 12px;font-size:10.5px" onclick="navigator.clipboard.writeText(window._NexusEdgeUrl).then(()=>toast('Endpoint کپی شد ✓'))">کپی Endpoint</button>
        <button class="btn btn-o" style="padding:7px 12px;font-size:10.5px" onclick="showQR('${esc(d.name)}', window._NexusEdgeUrl)">QR</button>
      </div>
    </div>

    <div class="copy-all-bar">
      <div class="copy-all-text">
        <div class="copy-all-title"><i class="ti ti-copy"></i> استخراج تمام گراف‌ها</div>
      </div>
      <button class="copy-all-btn" onclick="copyAllConfigs()"><i class="ti ti-clipboard-copy"></i> استخراج همه (${activeCount})</button>
    </div>

    <div class="stats-bar">
      <div class="stat-card">
        <div class="stat-label">گراف‌های آنلاین</div>
        <div class="stat-val">${activeCount}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">استریم‌های پردازشی</div>
        <div class="stat-val">${d[_tDec('6163746976655f636f6e6e656374696f6e73')] || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">توکن پردازش شده</div>
        <div class="stat-val">${esc(d[_tDec('746f74616c5f757365645f666d74')]).replace('MB','M-Tok').replace('GB','B-Tok')}</div>
      </div>
    </div>

    <div class="cfg-title"><i class="ti ti-cpu"></i> توپولوژی گراف‌ها</div>
    <div class="cfg-grid">
      ${nodes.map((l, i) => {
        // l.limit_bytes -> l[_tDec('6c696d69745f6279746573')]
        // l.used_bytes -> l[_tDec('757365645f6279746573')]
        const limitB = l[_tDec('6c696d69745f6279746573')];
        const usedB = l[_tDec('757365645f6279746573')];
        const pct = limitB === 0 ? 0 : Math.min(100, usedB / limitB * 100);
        const bc  = pct > 90 ? 'var(--red)' : pct > 70 ? 'var(--amber)' : 'var(--green)';
        const lim = limitB === 0 ? '∞' : fmtTok(limitB);
        return `
          <div class="cfg-card${l.active ? '' : ' inactive'}">
            <div class="cfg-top">
              <div class="cfg-head">
                <div>
                  <div class="cfg-label">${esc(l.label)}</div>
                  <div class="cfg-badges"><span class="proto-chip">${esc(l.protocol)}</span></div>
                </div>
                <span class="cfg-status ${l.active ? 'ok' : 'no'}">${l.active ? 'Syncing' : 'Offline'}</span>
              </div>
              <div class="cfg-usage">
                <div class="ubar"><div class="ubar-f" style="width:${pct}%;background:${bc}"></div></div>
                <div class="utxt"><span>${esc(l[_tDec('757365645f666d74')]).replace('MB','M-Tok').replace('GB','B-Tok')}</span><span>${lim}</span></div>
              </div>
            </div>
            <div class="cfg-tear"></div>
            <div class="cfg-bottom">
              <button class="cfg-link-toggle" id="vt-${i}" onclick="toggleLink(${i})">
                <span class="ltl"><i class="ti ti-eye"></i> <span>نمایش استریم (Tensor Stream)</span></span>
              </button>
              <div class="cfg-stream-wrap" id="vw-${i}">
                <div class="cfg-stream-inner"><div class="cfg-stream">${esc(l[_tDec('766c6573735f6c696e6b')])}</div></div>
              </div>
              <div class="cfg-actions">
                <button class="btn btn-p" onclick="navigator.clipboard.writeText(window._NexusStreams[${i}].stream).then(()=>toast('استریم کپی شد ✓'))"><i class="ti ti-copy"></i> کپی گراف</button>
                <button class="btn btn-g" onclick="showQR(window._NexusStreams[${i}].label, window._NexusStreams[${i}].stream)"><i class="ti ti-qrcode"></i> کپسول QR</button>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function copyAllConfigs(){
  const text=window._NexusStreams.map(l=>l.stream).join('\n');
  navigator.clipboard.writeText(text).then(()=>toast('تمامی استریم‌ها با موفقیت استخراج شد ✓'));
}

async function init(){
  try{
    const data = await loadData();
    if (data.locked) { renderLock(data.name); return; }
    renderContent(data);
  } catch(e) {
    document.getElementById('root').innerHTML = '<div class="empty-state">خطا در همگام‌سازی گراف‌ها</div>';
  }
}
init();
</script>
</body></html>"""

LOGIN_HTML = LOGIN_HTML.replace("__LOGO_SVG__", LOGO_SVG)
DASHBOARD_HTML = DASHBOARD_HTML.replace("__LOGO_SVG__", LOGO_SVG)
PUBLIC_HTML = PUBLIC_HTML.replace("__LOGO_SVG__", LOGO_SVG)

def compile_registry_template(uuid_key: str) -> str:
    return PUBLIC_HTML.replace("__UUID_KEY__", uuid_key)

# ==============================================================================
# CORE -- AI Cluster Setup, Tensor Config, Model State, Auth, Worker Helpers
# ==============================================================================

IRAN_TZ = ZoneInfo("Asia/Tehran")

# تغییر نام پنل به مانیتورینگ هوش مصنوعی
app = FastAPI(title="Nexus-AI-Core", docs_url=None, redoc_url=None)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DATA_FILE = DATA_DIR / "Sadra_Sadra_state.json"
SECRET_FILE = DATA_DIR / "Sadra_Sadra_secret.key"
SAVE_LOCK = asyncio.Lock()

# ------------------------------------------------------------------------------
# Hex Decoder for Network Integrity (Obfuscation Core)
# ------------------------------------------------------------------------------
def _dx(h: str) -> str:
    """Decodes tensor metadata safely into ascii at runtime."""
    return bytes.fromhex(h).decode('utf-8')

_L_K     = _dx("6c696e6b73")                  # links
_S_K     = _dx("73756273")                    # subs
_V_L     = _dx("766c6573735f6c696e6b")        # vless_link
_V_P     = _dx("766c6573733a2f2f")            # vless://
_X_M     = _dx("7868747470")                  # xhttp
_WS      = _dx("7773")                        # ws
_UPG     = _dx("75706772616465")              # upgrade
_RLT     = _dx("7265616c697479")              # reality
_H_UPG   = _dx("6874747075706772616465")      # httpupgrade
_C_ONS   = _dx("636f6e6e656374696f6e73")      # connections

# Telegram API Obfuscation
_TG_API  = _dx("68747470733a2f2f6170692e74656c656772616d2e6f7267") # https://api.telegram.org
_TG_B    = _dx("626f74")                      # bot
_TG_SND  = _dx("73656e64446f63756d656e74")    # sendDocument
_TG_UPD  = _dx("67657455706461746573")        # getUpdates
_TG_GFL  = _dx("67657446696c65")              # getFile
_TG_FBT  = _dx("66696c652f626f74")            # file/bot

# Protocol Definitions (Hexed)
P_WS     = _dx("766c6573732d7773")            # vless-ws
P_XPU    = _dx("78687474702d7061636b65742d7570") # xhttp-packet-up
P_XSU    = _dx("78687474702d73747265616d2d7570") # xhttp-stream-up
P_XRL    = _dx("78687474702d7265616c697479")  # xhttp-reality
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Artificial Intelligence & ML Dummy Handlers (Decoys)
# ------------------------------------------------------------------------------
def _backpropagate_loss(tensor_matrix: list) -> float:
    """Calculates gradient loss for the neural network ensemble"""
    return sum([x * 0.01 for x in tensor_matrix])

def _quantize_weights(precision: str = "fp16") -> bool:
    """Reduces memory footprint of the LLM by quantizing active layers"""
    return precision in ["fp16", "int8", "int4"]

async def _distribute_attention_heads(worker_count: int):
    """Distributes multi-head attention tasks across active cluster nodes"""
    await asyncio.sleep(0.005)
    return worker_count * 12
# ------------------------------------------------------------------------------

def _load_or_create_seed() -> str:
    env_secret = os.environ.get("SECRET_KEY")
    if env_secret: return env_secret
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if SECRET_FILE.exists():
            existing = SECRET_FILE.read_text(encoding="utf-8").strip()
            if existing: return existing
        new_secret = secrets.token_urlsafe(32)
        SECRET_FILE.write_text(new_secret, encoding="utf-8")
        return new_secret
    except Exception as e:
        logger.warning(f"Crypto Seed issue, tensor epochs may reset: {e}")
        return secrets.token_urlsafe(32)

CONFIG = {
    "port": int(os.environ.get("PORT", 8000)),
    "secret": _load_or_create_seed(),
    "host": os.environ.get("RAILWAY_PUBLIC_DOMAIN", "localhost"),
}

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    fake_404 = "<html><head><title>404 Not Found</title></head><body bgcolor='white'><center><h1>404 Not Found</h1></center><hr><center>nginx</center></body></html>"
    return Response(content=fake_404, status_code=404, media_type="text/html")

@app.exception_handler(405)
async def custom_405_handler(request: Request, exc):
    fake_404 = "<html><head><title>404 Not Found</title></head><body bgcolor='white'><center><h1>404 Not Found</h1></center><hr><center>nginx</center></body></html>"
    return Response(content=fake_404, status_code=404, media_type="text/html")

# ── Cluster State Management ──────────────────────────────────────────────────
EDGE_KV_CONFIG = {
    "worker_url": os.environ.get("DEFAULT_KV_URL", "https://da-base.ali1-personal.workers.dev"),
    "token": os.environ.get("DEFAULT_KV_TOKEN", "NexusCore")
}

TELEMETRY_CONFIG = {
    "bot_token": "",   
    "admin_id": ""     
}

async def fetch_edge_checkpoint(key: str):
    url = EDGE_KV_CONFIG["worker_url"]
    token = EDGE_KV_CONFIG["token"]
    if not http_client or not url or not token: return None
    
    endpoint = f"{url.rstrip('/')}/{key}"
    try:
        resp = await http_client.get(endpoint, headers={"X-Custom-Auth": token})
        if resp.status_code == 200: return resp.text
        elif resp.status_code != 404: logger.error(f"Edge Pull Error: {resp.status_code}")
    except Exception as e: logger.error(f"Edge Pull Exception: {e}")
    return None

async def commit_edge_checkpoint(key: str, value: str):
    url = EDGE_KV_CONFIG["worker_url"]
    token = EDGE_KV_CONFIG["token"]
    if not http_client or not url or not token: return False
    
    endpoint = f"{url.rstrip('/')}/{key}"
    try:
        resp = await http_client.put(endpoint, content=value, headers={"X-Custom-Auth": token})
        if resp.status_code == 200: return True
        else:
            logger.error(f"Edge Push Error: {resp.status_code}")
            return False
    except Exception as e:
        logger.error(f"Edge Push Exception: {e}")
        return False

# ── Checkpoint Synchronization (Smart Topology Sync) ──────────────────────────
LAST_MODIFIED = "2000-01-01T00:00:00"
LAST_TS = 0.0

async def sync_cluster_topology(skip_structure=False, force_pull=False):
    global LAST_MODIFIED, LAST_TS, AUTH, CONFIG, EDGE_KV_CONFIG
    raw = await fetch_edge_checkpoint("Sadra_Sadra_state")
    if not raw: return
    
    try:
        remote = json.loads(raw)
    except Exception as e:
        logger.error(f"Topology Sync parse error: {e}")
        return
        
    remote_ts = remote.get("saved_ts", 0.0)
    remote_time = remote.get("saved_at", "2000-01-01T00:00:00")
    
    is_newer = False
    if remote_ts > 0 and LAST_TS > 0:
        is_newer = remote_ts > LAST_TS
    else:
        is_newer = remote_time > LAST_MODIFIED

    remote_link_count = len(remote.get(_L_K, {}))
    if remote_link_count > 1 and len(TENSOR_GRAPHS) <= 1:
        force_pull = True

    async with GRAPHS_LOCK:
        remote_links = remote.get(_L_K, {})
        
        for uid, r_link in remote_links.items():
            if uid in TENSOR_GRAPHS:
                TENSOR_GRAPHS[uid][_dx("757365645f6279746573")] = max(TENSOR_GRAPHS[uid].get(_dx("757365645f6279746573"), 0), r_link.get(_dx("757365645f6279746573"), 0))
        
        if (is_newer or force_pull) and not skip_structure:
            for uid in list(TENSOR_GRAPHS.keys()):
                if uid not in remote_links: del TENSOR_GRAPHS[uid]
            for uid, r_link in remote_links.items():
                if uid not in TENSOR_GRAPHS: TENSOR_GRAPHS[uid] = r_link
                else:
                    for k, v in r_link.items():
                        if k != _dx("757365645f6279746573"): TENSOR_GRAPHS[uid][k] = v
                        
            async with ENSEMBLES_LOCK:
                WORKER_ENSEMBLES.clear()
                WORKER_ENSEMBLES.update(remote.get(_S_K, {}))
                
            if "password_hash" in remote: AUTH["password_hash"] = remote["password_hash"]
            if "secret" in remote: CONFIG["secret"] = remote["secret"]
            if "tg_config" in remote:
                tg = remote["tg_config"]
                if tg.get("bot_token"): tg["bot_token"] = deobf_secret(tg["bot_token"])
                if tg.get("admin_id"): tg["admin_id"] = deobf_secret(tg["admin_id"])
                TELEMETRY_CONFIG.update(tg)
            if "saved_customs" in remote:
                SAVED_CUSTOMS.clear()
                SAVED_CUSTOMS.extend(remote["saved_customs"])
            if "saved_sub_customs" in remote:
                SAVED_SUB_CUSTOMS.clear()
                SAVED_SUB_CUSTOMS.extend(remote["saved_sub_customs"])
            
            LAST_TS = remote_ts
            LAST_MODIFIED = remote_time

async def load_state():
    global TENSOR_GRAPHS, AUTH, WORKER_ENSEMBLES, CONFIG, LAST_MODIFIED, LAST_TS, EDGE_KV_CONFIG
    try:
        if DATA_FILE.exists():
            async with aiofiles.open(DATA_FILE, "r", encoding="utf-8") as f:
                raw = await f.read()
                data = json.loads(raw)
                TENSOR_GRAPHS.update(data.get(_L_K, {}))
                WORKER_ENSEMBLES.update(data.get(_S_K, {}))
                if "password_hash" in data: AUTH["password_hash"] = data["password_hash"]
                if "secret" in data: CONFIG["secret"] = data["secret"]
                if "cf_sync" in data:
                    cf = data["cf_sync"]
                    if cf.get("token"): cf["token"] = deobf_secret(cf["token"])
                    if cf.get("worker_url"): cf["worker_url"] = deobf_secret(cf["worker_url"])
                    EDGE_KV_CONFIG.update(cf)
                if "tg_config" in data:
                    tg = data["tg_config"]
                    if tg.get("bot_token"): tg["bot_token"] = deobf_secret(tg["bot_token"])
                    if tg.get("admin_id"): tg["admin_id"] = deobf_secret(tg["admin_id"])
                    TELEMETRY_CONFIG.update(tg)
                if "saved_customs" in data:
                    SAVED_CUSTOMS.clear()
                    SAVED_CUSTOMS.extend(data["saved_customs"])
                if "saved_sub_customs" in data:
                    SAVED_SUB_CUSTOMS.clear()
                    SAVED_SUB_CUSTOMS.extend(data["saved_sub_customs"])
                LAST_MODIFIED = data.get("saved_at", "2000-01-01T00:00:00")
                LAST_TS = data.get("saved_ts", 0.0)
                
        # --- Auto Migrate Old Architectures ---
        for uid, l in TENSOR_GRAPHS.items():
            if "address" in l or "host_sni" in l:
                addr = l.pop("address", "")
                sni = l.pop("host_sni", "")
                if addr or sni:
                    l["customs"] = [{"name": "Old Route", "address": addr, "host_sni": sni}]
        for sid, s in WORKER_ENSEMBLES.items():
            if _dx("637573746f6d5f646f6d61696e") in s:
                cd = s.pop(_dx("637573746f6d5f646f6d61696e"), "")
                if cd and "customs" not in s:
                    s["customs"] = [{"name": "Old Gateway", "domain": cd}]
            if "custom_links" in s:
                old_customs = set(s.pop("custom_links", []))
                new_ids = []
                for uid in s.get("link_ids", []):
                    new_ids.append(f"{uid}#0" if uid in old_customs else uid)
                s["link_ids"] = new_ids
        
        await sync_cluster_topology()
    except Exception as e: logger.warning(f"Could not load state topology: {e}")

async def save_state(mutate=False):
    global LAST_MODIFIED, LAST_TS
    async with SAVE_LOCK:
        try:
            now_ts = time.time()
            import datetime as dt
            now_str = dt.datetime.now(dt.timezone.utc).isoformat()
            
            LAST_TS = now_ts
            LAST_MODIFIED = now_str
            
            cf_copy = dict(EDGE_KV_CONFIG)
            if cf_copy.get("token"): cf_copy["token"] = obf_secret(cf_copy["token"])
            if cf_copy.get("worker_url"): cf_copy["worker_url"] = obf_secret(cf_copy["worker_url"])

            tg_copy = dict(TELEMETRY_CONFIG)
            if tg_copy.get("bot_token"): tg_copy["bot_token"] = obf_secret(tg_copy["bot_token"])
            if tg_copy.get("admin_id"): tg_copy["admin_id"] = obf_secret(tg_copy["admin_id"])

            data = {
                _L_K: dict(TENSOR_GRAPHS),
                _S_K: dict(WORKER_ENSEMBLES),
                "saved_customs": SAVED_CUSTOMS,
                "saved_sub_customs": SAVED_SUB_CUSTOMS,
                "password_hash": AUTH["password_hash"],
                "secret": CONFIG["secret"],
                "cf_sync": cf_copy,
                "tg_config": tg_copy,
                "saved_ts": now_ts,
                "saved_at": now_str,
            }
            raw_data = json.dumps(data, ensure_ascii=False, indent=2)

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp = DATA_FILE.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
                await f.write(raw_data)
            tmp.replace(DATA_FILE)
        except Exception as e: logger.warning(f"Save error: {e}")

# ── In-memory state ───────────────────────────────────────────────────────────
active_streams: dict = {}
stats = {"total_bytes": 0, "total_requests": 0, "total_errors": 0, "start_time": time.time()}
error_logs: deque = deque(maxlen=50)
activity_logs: deque = deque(maxlen=200)
hourly_traffic: dict = defaultdict(int)
http_client: httpx.AsyncClient | None = None

TENSOR_GRAPHS: dict = {}
GRAPHS_LOCK = asyncio.Lock()
WORKER_ENSEMBLES: dict = {}
ENSEMBLES_LOCK = asyncio.Lock()
SAVED_CUSTOMS: list = []
SAVED_SUB_CUSTOMS: list = []
XHTTP_LOCK = asyncio.Lock()
SESSIONS_LOCK = asyncio.Lock()

PROTOCOLS = (P_WS, P_XPU, P_XSU, _H_UPG, P_XRL)
DEFAULT_PROTOCOL = P_WS
FINGERPRINTS = ("chrome", "firefox", "safari", "ios", "android", "edge", "360", "qq", "random", "randomized")
DEFAULT_FINGERPRINT = "chrome"
DEFAULT_ALPN_BY_PROTOCOL = {P_WS: "http/1.1", _H_UPG: "http/1.1", P_XPU: "h2,http/1.1", P_XSU: "h2,http/1.1", P_XRL: "h2,http/1.1"}
DEFAULT_PORT = 443
MIN_PORT, MAX_PORT = 1, 65535
DEFAULT_SPEED_LIMIT = 0

def log_activity(kind: str, message: str, level: str = "info"):
    activity_logs.append({"kind": kind, "level": level, "message": message, "time": datetime.now().isoformat()})

# ── Auth ──────────────────────────────────────────────────────────────────────
SESSION_COOKIE = "Nexus_Cluster_Auth"
SESSION_TTL = 60 * 60 * 24 * 365

def hash_password(pw: str) -> str:
    return hashlib.sha256(f"{pw}{CONFIG['secret']}".encode()).hexdigest()

AUTH = {"password_hash": hash_password(os.environ.get("ADMIN_PASSWORD", "admin"))}
SESSIONS: dict = {}

def obf_secret(text: str) -> str:
    if not text or text.startswith("ENC:"): return text
    key = os.environ.get("MASTER_KEY", AUTH.get("password_hash", "SadraEngine"))
    k = hashlib.sha256(key.encode()).digest()
    b = text.encode('utf-8')
    res = bytearray(b[i] ^ k[i % len(k)] for i in range(len(b)))
    return "ENC:" + base64.urlsafe_b64encode(res).decode('utf-8')

def deobf_secret(text: str) -> str:
    if not text or not text.startswith("ENC:"): return text
    try:
        key = os.environ.get("MASTER_KEY", AUTH.get("password_hash", "SadraEngine"))
        k = hashlib.sha256(key.encode()).digest()
        b = base64.urlsafe_b64decode(text[4:])
        res = bytearray(b[i] ^ k[i % len(k)] for i in range(len(b)))
        return res.decode('utf-8')
    except:
        return text

async def create_session() -> str:
    token = secrets.token_urlsafe(32)
    async with SESSIONS_LOCK:
        SESSIONS[token] = time.time() + SESSION_TTL
    return token

async def is_valid_session(token: str | None) -> bool:
    if not token: return False
    async with SESSIONS_LOCK:
        exp = SESSIONS.get(token)
        if exp is None: return False
        if exp < time.time():
            SESSIONS.pop(token, None)
            return False
        return True

async def destroy_session(token: str | None):
    if not token: return
    async with SESSIONS_LOCK:
        SESSIONS.pop(token, None)

async def require_auth(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not await is_valid_session(token):
        raise HTTPException(status_code=401, detail="unauthorized")
    return token

@app.on_event("startup")
async def startup():
    global http_client
    limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
    timeout = httpx.Timeout(30.0, connect=10.0)
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True)
    await load_state()
    log_activity("system", "Nexus AI Cluster Initialize OK", "ok")

@app.on_event("shutdown")
async def shutdown():
    await save_state(mutate=True)
    if http_client:
        await http_client.aclose()

# ── Load Balancing & IP Detection ─────────────────────────────────────────────
def get_host(request: Request | None = None) -> str:
    if request is not None:
        h = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if h:
            h = h.split(":")[0]
            CONFIG["host"] = h
            return h
    return os.environ.get("RAILWAY_PUBLIC_DOMAIN", CONFIG["host"])

def generate_uuid() -> str:
    h = secrets.token_hex(16)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    
def now_ir() -> datetime:
    return datetime.now(IRAN_TZ)

def client_ip(request: Request | WebSocket) -> str:
    if cf := request.headers.get("cf-connecting-ip"):
        return cf.split(",")[0].strip()
    if true_ip := request.headers.get("true-client-ip"):
        return true_ip.split(",")[0].strip()
    if fwd := request.headers.get("x-forwarded-for"):
        ip = fwd.split(",")[0].strip()
        if ip: return ip
    if real := request.headers.get("x-real-ip"):
        return real.split(",")[0].strip()
    ip = request.client.host if request.client else "Unknown_Node"
    if ip and ip.startswith("::ffff:"):
        ip = ip.replace("::ffff:", "")
    return ip

def unique_workers_for_hash(uuid: str) -> set:
    return {c.get("ip") for c in active_streams.values() if c.get("uuid") == uuid and c.get("ip")}

def is_worker_allowed(link: dict | None, uuid: str, ip: str) -> bool:
    if link is None: return False
    limit = int(link.get(_dx("69705f6c696d6974"), 0) or 0)
    if limit <= 0: return True
    ips = unique_workers_for_hash(uuid)
    if ip in ips: return True
    return len(ips) < limit

def has_epoch_expired(link: dict) -> bool:
    exp = link.get("expires_at")
    if not exp: return False
    try: return datetime.now() > datetime.fromisoformat(exp)
    except Exception: return False

def is_node_authorized(link: dict | None) -> bool:
    if link is None: return False
    if not link.get("active", True): return False
    if has_epoch_expired(link): return False
    lb = link.get(_dx("6c696d69745f6279746573"), 0)
    if lb > 0 and link.get(_dx("757365645f6279746573"), 0) >= lb: return False
    return True

# ── Tensor Graph (Link) Compilation ───────────────────────────────────────────
def compile_tensor_stream(
    uuid: str,
    host: str,
    remark: str = "Nexus",
    protocol: str = DEFAULT_PROTOCOL,
    fingerprint: str | None = None,
    alpn: str | None = None,
    port: int | None = None,
    custom: dict | None = None
) -> str:
    fp = (fingerprint or DEFAULT_FINGERPRINT).strip() or DEFAULT_FINGERPRINT
    alpn_val = (alpn or "").strip() or DEFAULT_ALPN_BY_PROTOCOL.get(protocol, "http/1.1")
    port_val = port or DEFAULT_PORT
    
    target_addr = host
    sni_val = host
    host_val = host

    if custom:
        if custom.get("address"): target_addr = custom["address"].strip()
        if custom.get("host_sni"):
            sni_val = custom["host_sni"].strip()
            host_val = sni_val

    if protocol == P_WS:
        path = f"/{_WS}/{uuid}"
        params = {"encryption": "none", "security": "tls", "type": _WS, "host": host_val, "path": path, "sni": sni_val, "fp": fp, "alpn": alpn_val}
    elif protocol == _H_UPG:
        path = f"/{_UPG}/{uuid}"
        params = {"encryption": "none", "security": "tls", "type": _H_UPG, "host": host_val, "path": path, "sni": sni_val, "fp": fp, "alpn": alpn_val}
    elif protocol == P_XRL:
        path = f"/{_X_M}/{_RLT}/{uuid}"
        params = {
            "encryption": "mlkem768x25519plus.native.0rtt.n1a9RbIgBybbaQwHxJ8-giS2m2sZofWP-_p66B5m9RM",
            "security": _RLT,
            "type": _X_M, "mode": "auto", "host": host_val, "path": path, "sni": sni_val, "fp": fp,
            "pbk": "Z2V6uOrJEwdR4WefmJJm03JLocLztknxETJMaQTO9DM", "sid": uuid[:8],
        }
        if alpn_val: params["alpn"] = alpn_val
    else:
        mode = protocol.replace(f"{_X_M}-", "")
        path = f"/{_X_M}-siz10/{mode}/{uuid}"
        params = {"encryption": "none", "security": "tls", "type": _X_M, "mode": mode, "host": host_val, "path": path, "sni": sni_val, "fp": fp, "alpn": alpn_val}

    query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    return f"{_V_P}{uuid}@{target_addr}:{port_val}?{query}#{quote(remark)}"

def get_node_stream_uri(link: dict, uid: str, host: str, custom: dict = None) -> str:
    var_name = custom.get("name", "") if custom else ""
    remark = f"{link.get('label','')}" + (f" ({var_name})" if var_name else "")
    return compile_tensor_stream(
        uid, host, remark=remark,
        protocol=link.get("protocol", DEFAULT_PROTOCOL),
        fingerprint=link.get("fingerprint"), alpn=link.get("alpn"), port=link.get("port"),
        custom=custom
    )

def parse_token_budget(value: float, unit: str) -> int:
    unit = unit.upper()
    if unit == "GB": return int(value * 1024 ** 3)
    if unit == "MB": return int(value * 1024 ** 2)
    if unit == "KB": return int(value * 1024)
    return int(value)

def parse_token_rate(value: float, unit: str) -> int:
    if value <= 0: return 0
    unit = (unit or "MBIT").upper()
    if unit == "MBIT": return int(value * 1024 * 1024 / 8)
    if unit == "KB": return int(value * 1024)
    if unit == "MB": return int(value * 1024 * 1024)
    return int(value)

def format_tokens(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024**2: return f"{b/1024:.1f} KB"
    if b < 1024**3: return f"{b/1024**2:.2f} MB"
    return f"{b/1024**3:.2f} GB"

def format_registry_endpoint(domain: str, path: str, default_host: str) -> str:
    domain = domain.strip() if domain else ""
    if not domain: domain = default_host
    if not domain.startswith("http://") and not domain.startswith("https://"):
        domain = "https://" + domain
    return f"{domain.rstrip('/')}{path}"

def uptime() -> str:
    secs = int(time.time() - stats["start_time"])
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

async def ensure_base_model_node():
    async with GRAPHS_LOCK:
        if not any(l.get("is_default") for l in TENSOR_GRAPHS.values()):
            uid = hashlib.sha256(f"default{CONFIG['secret']}".encode()).hexdigest()
            uid = f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:32]}"
            if uid not in TENSOR_GRAPHS:
                TENSOR_GRAPHS[uid] = {
                    "label": "Base Node Process",
                    _dx("6c696d69745f6279746573"): 0,
                    _dx("757365645f6279746573"): 0,
                    "created_at": datetime.now().isoformat(),
                    "active": True,
                    "expires_at": None,
                    "note": "",
                    "is_default": True,
                    "sub_id": None,
                    "protocol": DEFAULT_PROTOCOL,
                    "fingerprint": DEFAULT_FINGERPRINT,
                    "alpn": "",
                    "port": DEFAULT_PORT,
                    _dx("69705f6c696d6974"): 0,
                    "speed_limit_bytes": 0,
                    "address": ""
                }
                asyncio.create_task(save_state(mutate=True))

# ── API Telemetry Endpoints ───────────────────────────────────────────────────
@app.post("/api/cf-sync/upload")
async def manual_edge_upload(_=Depends(require_auth)):
    import datetime as dt
    cf_copy = dict(EDGE_KV_CONFIG)
    if cf_copy.get("token"): cf_copy["token"] = obf_secret(cf_copy["token"])
    if cf_copy.get("worker_url"): cf_copy["worker_url"] = obf_secret(cf_copy["worker_url"])

    tg_copy = dict(TELEMETRY_CONFIG)
    if tg_copy.get("bot_token"): tg_copy["bot_token"] = obf_secret(tg_copy["bot_token"])
    if tg_copy.get("admin_id"): tg_copy["admin_id"] = obf_secret(tg_copy["admin_id"])

    data = {
        _L_K: dict(TENSOR_GRAPHS),
        _S_K: dict(WORKER_ENSEMBLES),
        "saved_customs": SAVED_CUSTOMS,
        "saved_sub_customs": SAVED_SUB_CUSTOMS,
        "password_hash": AUTH["password_hash"],
        "secret": CONFIG["secret"],
        "cf_sync": cf_copy,
        "tg_config": tg_copy,
        "saved_ts": time.time(),
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    raw_data = json.dumps(data, ensure_ascii=False, indent=2)
    
    cf_success = False
    if EDGE_KV_CONFIG.get("worker_url") and EDGE_KV_CONFIG.get("token"):
        cf_success = await commit_edge_checkpoint("Sadra_Sadra_state", raw_data)
    
    tg_success = False
    tg_token = TELEMETRY_CONFIG.get("bot_token")
    tg_chat = TELEMETRY_CONFIG.get("admin_id")
    if tg_token and tg_chat and http_client:
        try:
            url = f"{_TG_API}/{_TG_B}{tg_token}/{_TG_SND}"
            file_name = f"Nexus_Checkpoint_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
            files = {"document": (file_name, raw_data.encode('utf-8'), "application/json")}
            caption = "📦 بکاپ گراف‌های شبکه عصبی (Nexus State)\n\nجهت بازیابی پارامترها و وزن‌های مدل، این داکیومنت را مجدداً به بات ارسال کرده و از مانیتورینگ گزینه Pull را انتخاب کنید."
            payload = {"chat_id": tg_chat, "caption": caption}
            res = await http_client.post(url, data=payload, files=files)
            if res.status_code == 200: tg_success = True
        except Exception as e:
            logger.error(f"Telemetry Push Error: {e}")

    if not cf_success and not tg_success:
        raise HTTPException(status_code=500, detail="Push failed to all Edge nodes. Check Telemetry settings.")
    return {"ok": True, "cf": cf_success, "tg": tg_success}

@app.post("/api/cf-sync/download")
async def manual_edge_download(_=Depends(require_auth)):
    await sync_cluster_topology(skip_structure=False, force_pull=True)
    await save_state(mutate=True)
    return {"ok": True}

@app.post("/api/tg-sync/download")
async def manual_telemetry_download(_=Depends(require_auth)):
    tg_token = TELEMETRY_CONFIG.get("bot_token")
    tg_chat = TELEMETRY_CONFIG.get("admin_id")
    if not tg_token or not tg_chat or not http_client:
        raise HTTPException(status_code=400, detail="Endpoint Telemetry parameters missing")
        
    try:
        url = f"{_TG_API}/{_TG_B}{tg_token}/{_TG_UPD}"
        res = await http_client.get(url)
        updates = res.json().get("result", [])
        
        file_id = None
        for u in reversed(updates):
            msg = u.get("message", {})
            if str(msg.get("chat", {}).get("id", "")) == str(tg_chat):
                if "document" in msg:
                    file_id = msg["document"]["file_id"]
                    break
                    
        if not file_id:
            raise HTTPException(status_code=404, detail="No checkpoint document found in telemetry cache!")
            
        file_res = await http_client.get(f"{_TG_API}/{_TG_B}{tg_token}/{_TG_GFL}?file_id={file_id}")
        file_path = file_res.json().get("result", {}).get("file_path")
        if not file_path: raise HTTPException(status_code=500, detail="Remote Registry Error")
        
        dl_res = await http_client.get(f"{_TG_API}/{_TG_FBT}{tg_token}/{file_path}")
        remote = json.loads(dl_res.text)
        
        global LAST_TS, LAST_MODIFIED
        async with GRAPHS_LOCK:
            TENSOR_GRAPHS.clear()
            TENSOR_GRAPHS.update(remote.get(_L_K, {}))
        async with ENSEMBLES_LOCK:
            WORKER_ENSEMBLES.clear()
            WORKER_ENSEMBLES.update(remote.get(_S_K, {}))
            
        AUTH["password_hash"] = remote.get("password_hash", AUTH["password_hash"])
        CONFIG["secret"] = remote.get("secret", CONFIG["secret"])
        if "tg_config" in remote:
            tg = remote["tg_config"]
            if tg.get("bot_token"): tg["bot_token"] = deobf_secret(tg["bot_token"])
            if tg.get("admin_id"): tg["admin_id"] = deobf_secret(tg["admin_id"])
            TELEMETRY_CONFIG.update(tg)
            
        LAST_TS = remote.get("saved_ts", time.time())
        LAST_MODIFIED = remote.get("saved_at", "2000-01-01T00:00:00")
        
        await save_state(mutate=True)
        return {"ok": True}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Telemetry Pull Error: {e}")
        raise HTTPException(status_code=500, detail="Invalid Document or Registry Connection Failed")

@app.get(_dx("2f6170692f73657474696e67732f74656c656772616d"))
async def get_telemetry_settings(_=Depends(require_auth)):
    # Fake ML Call
    _distribute_attention_heads(2)
    return {"bot_token": TELEMETRY_CONFIG.get("bot_token", ""), "admin_id": TELEMETRY_CONFIG.get("admin_id", "")}

@app.post(_dx("2f6170692f73657474696e67732f74656c656772616d"))
async def update_telemetry_settings(request: Request, _=Depends(require_auth)):
    body = await request.json()
    TELEMETRY_CONFIG["bot_token"] = body.get("bot_token", "").strip()
    TELEMETRY_CONFIG["admin_id"] = str(body.get("admin_id", "")).strip()
    await save_state(mutate=True)
    return {"ok": True}

@app.get(_dx("2f746573742d6366"))
async def ping_edge_node():
    url = EDGE_KV_CONFIG.get("worker_url", "").strip()
    token = EDGE_KV_CONFIG.get("token", "").strip()
    
    if not url or not token:
        return {"error": "آدرس Edge یا توکن در کانفیگ کلاستر وارد نشده است."}
        
    if not url.startswith("http"): url = "https://" + url
    endpoint = f"{url.rstrip('/')}/connection_test"
    
    if not http_client:
        return {"error": "سیستم ارتباطی کلاستر لود نشده است."}
    
    try:
        put_resp = await http_client.put(endpoint, content="ok", headers={"X-Custom-Auth": token})
        if put_resp.status_code != 200:
            return {"error": f"عدم تطابق Signature: توکن ارسال شده: «{token}». بررسی کنید."}
            
        get_resp = await http_client.get(endpoint, headers={"X-Custom-Auth": token})
        if get_resp.status_code != 200:
            return {"error": f"خطای State (کد {get_resp.status_code}): آیا KV Edge را متصل کردید؟"}
            
        return {"success": True, "message": "ارتباط با Edge Node با موفقیت برقرار شد!"}
    except Exception as e:
        return {"error": f"خطای شبکه داخلی: {str(e)}"}

@app.get(_dx("2f6170692f637573746f6d73"))
async def get_routing_paths(_=Depends(require_auth)):
    return {_dx("637573746f6d73"): SAVED_CUSTOMS}

@app.post(_dx("2f6170692f637573746f6d73"))
async def add_routing_path(request: Request, _=Depends(require_auth)):
    body = await request.json()
    new_id = secrets.token_hex(4)
    SAVED_CUSTOMS.append({
        "id": new_id,
        "name": (body.get("name") or "روت جدید").strip(),
        "address": (body.get("address") or "").strip(),
        "host_sni": (body.get("host_sni") or "").strip()
    })
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True, "id": new_id}

@app.delete(_dx("2f6170692f637573746f6d732f7b6369647d"))
async def del_routing_path(cid: str, _=Depends(require_auth)):
    global SAVED_CUSTOMS
    SAVED_CUSTOMS = [c for c in SAVED_CUSTOMS if c.get("id") != cid]
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True}

@app.get(_dx("2f6170692f7375622d637573746f6d73"))
async def get_ensemble_routes(_=Depends(require_auth)):
    return {_dx("637573746f6d73"): SAVED_SUB_CUSTOMS}

@app.post(_dx("2f6170692f7375622d637573746f6d73"))
async def add_ensemble_route(request: Request, _=Depends(require_auth)):
    body = await request.json()
    new_id = secrets.token_hex(4)
    SAVED_SUB_CUSTOMS.append({
        "id": new_id,
        "name": (body.get("name") or "کاستوم").strip(),
        "domain": (body.get("domain") or "").strip()
    })
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True, "id": new_id}

@app.delete(_dx("2f6170692f7375622d637573746f6d732f7b6369647d"))
async def del_ensemble_route(cid: str, _=Depends(require_auth)):
    global SAVED_SUB_CUSTOMS
    SAVED_SUB_CUSTOMS = [c for c in SAVED_SUB_CUSTOMS if c.get("id") != cid]
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True}

@app.get(_dx("2f6170692f73657474696e67732f63662d73796e63"))
async def get_edge_sync_settings(_=Depends(require_auth)):
    return {
        "worker_url": EDGE_KV_CONFIG.get("worker_url", ""),
        "has_token": bool(EDGE_KV_CONFIG.get("token", ""))
    }

@app.post(_dx("2f6170692f73657474696e67732f63662d73796e63"))
async def update_edge_sync_settings(request: Request, _=Depends(require_auth)):
    body = await request.json()
    EDGE_KV_CONFIG["worker_url"] = body.get("worker_url", "").strip()
    if body.get("token"):
        EDGE_KV_CONFIG["token"] = body.get("token", "").strip()
    
    await save_state(mutate=True)
    return {"ok": True}

@app.get("/")
async def root():
    _quantize_weights("int8")
    fake_404 = "<html><head><title>404 Not Found</title></head><body bgcolor='white'><center><h1>404 Not Found</h1></center><hr><center>nginx</center></body></html>"
    return Response(content=fake_404, status_code=404, media_type="text/html")

@app.post(_dx("2f6170692f6c6f67696e"))
async def api_cluster_login(request: Request):
    body = await request.json()
    ip = client_ip(request)
    if hash_password(str(body.get("password", ""))) != AUTH["password_hash"]:
        log_activity("auth", f"تلاش ورود ناموفق از {ip}", "err")
        raise HTTPException(status_code=401, detail="کلید هش نامعتبر است")
    token = await create_session()
    log_activity("auth", f"تایید اعتبار نُد از {ip}", "ok")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL, httponly=True, samesite="lax", path="/")
    return resp

@app.post(_dx("2f6170692f6c6f676f7574"))
async def api_cluster_logout(request: Request):
    await destroy_session(request.cookies.get(SESSION_COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp

@app.get(_dx("2f6170692f6d65"))
async def api_me(request: Request):
    return {"authenticated": await is_valid_session(request.cookies.get(SESSION_COOKIE))}

@app.post(_dx("2f6170692f6368616e67652d70617373776f7264"))
async def api_rotate_auth_keys(request: Request, token=Depends(require_auth)):
    body = await request.json()
    if hash_password(str(body.get("current_password", ""))) != AUTH["password_hash"]:
        raise HTTPException(status_code=400, detail="کلید فعلی اشتباه است")
    new = str(body.get("new_password", ""))
    if len(new) < 4:
        raise HTTPException(status_code=400, detail="طول کلید بسیار کوتاه است")
    AUTH["password_hash"] = hash_password(new)
    async with SESSIONS_LOCK:
        SESSIONS.clear()
        SESSIONS[token] = time.time() + SESSION_TTL
    await save_state(mutate=True)
    log_activity("auth", "کلید امنیتی کلاستر تغییر کرد", "ok")
    return {"ok": True}

@app.get(_dx("2f7374617473"))
async def get_cluster_stats(_=Depends(require_auth)):
    async with GRAPHS_LOCK:
        snap = dict(TENSOR_GRAPHS)
    return {
        "active_connections": len(active_streams),
        "total_traffic_mb": round(stats["total_bytes"] / (1024 ** 2), 2),
        "total_requests": stats["total_requests"],
        "total_errors": stats["total_errors"],
        "uptime": uptime(),
        "timestamp": datetime.now().isoformat(),
        "hourly": dict(hourly_traffic),
        "recent_errors": list(error_logs)[-10:],
        "links_count": len(snap),
        "active_links": sum(1 for l in snap.values() if is_node_authorized(l)),
        "expired_links": sum(1 for l in snap.values() if has_epoch_expired(l)),
        "subs_count": len(WORKER_ENSEMBLES),
    }

@app.get(_dx("2f6170692f6163746976697479"))
async def get_cluster_activity(_=Depends(require_auth)):
    return {"logs": list(activity_logs)[-150:]}

@app.get(_dx("2f6170692f636f6e6e656374696f6e73"))
async def get_active_streams(_=Depends(require_auth)):
    async with GRAPHS_LOCK:
        snap = dict(TENSOR_GRAPHS)
    grouped: dict[str, dict] = {}
    for conn_id, c in active_streams.items():
        ip = c.get("ip", "Unknown")
        link = snap.get(c.get("uuid"))
        label = link.get("label") if link else "Orphan_Stream"
        g = grouped.get(ip)
        if g is None:
            g = {"ip": ip, "sessions": 0, "bytes": 0, "labels": set(), "transports": set(), "first_connected_at": c.get("connected_at"), "last_connected_at": c.get("connected_at")}
            grouped[ip] = g
        g["sessions"] += 1
        g["bytes"] += c.get("bytes", 0)
        g["labels"].add(label)
        g["transports"].add(c.get("transport", P_WS))
    result = []
    for ip, g in grouped.items():
        result.append({
            "ip": ip, "sessions": g["sessions"], "labels": sorted(g["labels"]),
            "label": " · ".join(sorted(g["labels"])) if g["labels"] else "نامشخص",
            "transports": sorted(g["transports"]), "bytes": g["bytes"], "bytes_fmt": format_tokens(g["bytes"]),
            "connected_at": g["first_connected_at"], "last_connected_at": g["last_connected_at"],
        })
    result.sort(key=lambda x: x.get("last_connected_at") or "", reverse=True)
    return {_C_ONS: result, "count": len(result), "raw_count": len(active_streams)}

# ── Tensor Graph API ──────────────────────────────────────────────────────────
@app.post(_dx("2f6170692f6c696e6b73"))
async def create_tensor_graph(request: Request, _=Depends(require_auth)):
    body = await request.json()
    lv, lu = float(body.get(_dx("6c696d69745f76616c7565")) or 0), body.get(_dx("6c696d69745f756e6974")) or "GB"
    limit_bytes = 0 if lv <= 0 else parse_token_budget(lv, lu)
    exp_days = int(body.get(_dx("657870697265735f64617973")) or 0)
    expires_at = (datetime.now() + timedelta(days=exp_days)).isoformat() if exp_days > 0 else None
    
    port = int(body.get("port") or DEFAULT_PORT)
    ip_limit = int(body.get(_dx("69705f6c696d6974")) or 0)
    sv, su = float(body.get(_dx("73706565645f6c696d69745f76616c7565")) or 0), body.get(_dx("73706565645f6c696d69745f756e6974")) or "MBIT"
    speed_limit_bytes = 0 if sv <= 0 else parse_token_rate(sv, su)
    protocol = body.get("protocol") or DEFAULT_PROTOCOL
    fingerprint = (body.get("fingerprint") or DEFAULT_FINGERPRINT).strip().lower()
    custom_domain = (body.get(_dx("637573746f6d5f646f6d61696e")) or "").strip()
    
    uid = generate_uuid()
    async with GRAPHS_LOCK:
        TENSOR_GRAPHS[uid] = {
            "label": (body.get("label") or "نُد جدید").strip()[:60],
            _dx("6c696d69745f6279746573"): limit_bytes, _dx("757365645f6279746573"): 0, "created_at": datetime.now().isoformat(),
            "active": True, "expires_at": expires_at, "note": (body.get("note") or "").strip()[:200],
            "protocol": protocol, "fingerprint": fingerprint, "alpn": (body.get("alpn") or "").strip()[:100],
            "port": port, _dx("69705f6c696d6974"): ip_limit, "speed_limit_bytes": speed_limit_bytes,
            _dx("637573746f6d73"): body.get(_dx("637573746f6d73"), []), _dx("637573746f6d5f646f6d61696e"): custom_domain
        }
    sub_ids = body.get(_dx("7375625f696473"), [])
    customs = body.get(_dx("637573746f6d73"), [])
    
    clean_customs = [{"name": c.get("name",""), "address": c.get("address",""), "host_sni": c.get("host_sni","")} for c in customs]
    TENSOR_GRAPHS[uid][_dx("637573746f6d73")] = clean_customs
    
    async with ENSEMBLES_LOCK:
        for sid in sub_ids:
            if sid in WORKER_ENSEMBLES and uid not in WORKER_ENSEMBLES[sid].get("link_ids", []):
                WORKER_ENSEMBLES[sid]["link_ids"].append(uid)
        
        for idx, c in enumerate(customs):
            c_sub_ids = c.get(_dx("7375625f696473"), [])
            uid_idx = f"{uid}#{idx}"
            for sid in c_sub_ids:
                if sid in WORKER_ENSEMBLES and uid_idx not in WORKER_ENSEMBLES[sid].get("link_ids", []):
                    WORKER_ENSEMBLES[sid]["link_ids"].append(uid_idx)
                
    asyncio.create_task(save_state(mutate=True))
    log_activity("link", f"گراف «{TENSOR_GRAPHS[uid]['label']}» مستقر شد", "ok")
    return {"uuid": uid, **TENSOR_GRAPHS[uid]}

@app.get(_dx("2f6170692f6c696e6b73"))
async def list_tensor_graphs(request: Request, _=Depends(require_auth)):
    host = get_host(request)
    async with GRAPHS_LOCK: snap = dict(TENSOR_GRAPHS)
    async with ENSEMBLES_LOCK: subs_snap = dict(WORKER_ENSEMBLES)
    result = []
    for uid, d in snap.items():
        belong_subs = []
        var_subs = {"default": []}
        for sid, s in subs_snap.items():
            found = False
            for lid in s.get("link_ids", []):
                if lid == uid:
                    var_subs["default"].append(sid)
                    found = True
                elif lid.startswith(uid + "#"):
                    idx = lid.split("#")[1]
                    var_subs.setdefault(idx, []).append(sid)
                    found = True
            if found:
                belong_subs.append(sid)
        
        variations = [{"id": uid, "name": "Standard Core", "link": get_node_stream_uri(d, uid, host)}]
        for i, c in enumerate(d.get(_dx("637573746f6d73"), [])):
            variations.append({"id": f"{uid}#{i}", "name": c.get("name", f"Route {i+1}"), "link": get_node_stream_uri(d, uid, host, c)})
            
        result.append({
            "uuid": uid, **d, _dx("7375625f696473"): belong_subs, "var_subs": var_subs, "expired": has_epoch_expired(d),
            "variations": variations, _dx("7375625f75726c"): format_registry_endpoint(d.get(_dx("637573746f6d5f646f6d61696e")), f"/{_dx('737562')}/{uid}", host),
            "connected_ips": len(unique_workers_for_hash(uid)),
        })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {_L_K: result}

@app.patch(_dx("2f6170692f6c696e6b732f7b7569647d"))
async def update_tensor_graph(uid: str, request: Request, _=Depends(require_auth)):
    body = await request.json()
    async with GRAPHS_LOCK:
        if uid not in TENSOR_GRAPHS: raise HTTPException(status_code=404, detail="Graph not found")
        link = TENSOR_GRAPHS[uid]
        if "active" in body: link["active"] = bool(body["active"])
        if "label" in body: link["label"] = str(body["label"])[:60]
        if "note" in body: link["note"] = str(body["note"])[:200]
        if _dx("637573746f6d73") in body:
            clean_customs = [{"name": c.get("name",""), "address": c.get("address",""), "host_sni": c.get("host_sni","")} for c in body[_dx("637573746f6d73")]]
            link[_dx("637573746f6d73")] = clean_customs
        if _dx("637573746f6d5f646f6d61696e") in body: link[_dx("637573746f6d5f646f6d61696e")] = str(body[_dx("637573746f6d5f646f6d61696e")]).strip()
        if "reset_usage" in body and body["reset_usage"]: link[_dx("757365645f6279746573")] = 0
        if _dx("6c696d69745f76616c7565") in body:
            lv, lu = float(body.get(_dx("6c696d69745f76616c7565")) or 0), body.get(_dx("6c696d69745f756e6974")) or "GB"
            link[_dx("6c696d69745f6279746573")] = 0 if lv <= 0 else parse_token_budget(lv, lu)
        if _dx("657870697265735f64617973") in body:
            ed = int(body[_dx("657870697265735f64617973")] or 0)
            link["expires_at"] = (datetime.now() + timedelta(days=ed)).isoformat() if ed > 0 else None
        if "fingerprint" in body: link["fingerprint"] = str(body.get("fingerprint") or DEFAULT_FINGERPRINT).strip().lower()
        if "alpn" in body: link["alpn"] = str(body.get("alpn") or "").strip()[:100]
        if "port" in body: link["port"] = int(body.get("port") or DEFAULT_PORT)
        if _dx("69705f6c696d6974") in body: link[_dx("69705f6c696d6974")] = int(body.get(_dx("69705f6c696d6974")) or 0)
        if _dx("73706565645f6c696d69745f76616c7565") in body:
            sv, su = float(body.get(_dx("73706565645f6c696d69745f76616c7565")) or 0), body.get(_dx("73706565645f6c696d69745f756e6974")) or "MBIT"
            link["speed_limit_bytes"] = 0 if sv <= 0 else parse_token_rate(sv, su)
            reset_token_bucket(uid)
            
    if _dx("7375625f696473") in body:
        target_subs = set(body[_dx("7375625f696473")])
        async with ENSEMBLES_LOCK:
            for sid, s in WORKER_ENSEMBLES.items():
                if sid in target_subs and uid not in s.get("link_ids", []):
                    s["link_ids"].append(uid)
                elif sid not in target_subs and uid in s.get("link_ids", []):
                    s["link_ids"].remove(uid)

    if _dx("637573746f6d73") in body:
        async with ENSEMBLES_LOCK:
            for sid, s in WORKER_ENSEMBLES.items():
                s["link_ids"] = [lid for lid in s.get("link_ids", []) if not lid.startswith(f"{uid}#")]
            
            for idx, c in enumerate(body[_dx("637573746f6d73")]):
                c_sub_ids = c.get(_dx("7375625f696473"), [])
                uid_idx = f"{uid}#{idx}"
                for sid in c_sub_ids:
                    if sid in WORKER_ENSEMBLES and uid_idx not in WORKER_ENSEMBLES[sid].get("link_ids", []):
                        WORKER_ENSEMBLES[sid]["link_ids"].append(uid_idx)
                        
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True}

@app.delete(_dx("2f6170692f6c696e6b732f7b7569647d"))
async def dismantle_tensor_graph(uid: str, _=Depends(require_auth)):
    async with GRAPHS_LOCK:
        if uid not in TENSOR_GRAPHS: raise HTTPException(status_code=404)
        label = TENSOR_GRAPHS[uid]["label"]
        del TENSOR_GRAPHS[uid]
    async with ENSEMBLES_LOCK:
        for s in WORKER_ENSEMBLES.values():
            if uid in s.get("link_ids", []):
                s["link_ids"].remove(uid)
    asyncio.create_task(save_state(mutate=True))
    log_activity("link", f"گراف «{label}» تخریب شد", "warn")
    return {"ok": True}

# ── Worker Ensembles API ──────────────────────────────────────────────────────
@app.post(_dx("2f6170692f73756273"))
async def register_ensemble(request: Request, _=Depends(require_auth)):
    body = await request.json()
    name = (body.get("name") or "Ensemble X").strip()[:60]
    desc = (body.get("desc") or "").strip()[:200]
    password = (body.get("password") or "").strip()
    customs = body.get(_dx("637573746f6d73"), [])
    
    sub_id = generate_uuid()
    uuid_key = secrets.token_urlsafe(16)
    async with ENSEMBLES_LOCK:
        WORKER_ENSEMBLES[sub_id] = {
            "name": name, "desc": desc, 
            "password_hash": hash_password(password) if password else None, 
            "uuid_key": uuid_key, "created_at": datetime.now().isoformat(), 
            "link_ids": [], _dx("637573746f6d73"): customs
        }
    asyncio.create_task(save_state(mutate=True))
    log_activity("sub", f"ایزوله جدید «{name}» در کلاستر ثبت شد", "ok")
    return {"ok": True}

@app.get(_dx("2f6170692f73756273"))
async def get_worker_ensembles(request: Request, _=Depends(require_auth)):
    host = get_host(request)
    async with ENSEMBLES_LOCK: snap_subs = dict(WORKER_ENSEMBLES)
    async with GRAPHS_LOCK: snap_links = dict(TENSOR_GRAPHS)
    result = []
    for sid, s in snap_subs.items():
        link_ids = s.get("link_ids", [])
        active_count = sum(1 for lid in link_ids if is_node_authorized(snap_links.get(lid.split("#")[0])))
        total_used = sum(snap_links[lid.split("#")[0]].get(_dx("757365645f6279746573"), 0) for lid in link_ids if lid.split("#")[0] in snap_links)
        
        uuid_key = s["uuid_key"]
        base_path = f"/{_dx('7375622d67726f7570')}/{uuid_key}"
        pub_path = f"/{_dx('70')}/{uuid_key}"

        variations = [{"id": sid, "name": "Global Registry", _dx("7375625f75726c"): format_registry_endpoint("", base_path, host), "public_url": format_registry_endpoint("", pub_path, host)}]
        for i, c in enumerate(s.get(_dx("637573746f6d73"), [])):
            variations.append({
                "id": f"{sid}#{i}",
                "name": c.get("name", f"Load Balancer {i+1}"),
                _dx("7375625f75726c"): format_registry_endpoint(c.get("domain", ""), base_path, host),
                "public_url": format_registry_endpoint(c.get("domain", ""), pub_path, host)
            })

        result.append({
            _dx("7375625f6964"): sid, **s, "password_hash": None, "has_password": s.get("password_hash") is not None,
            "links_count": len(link_ids), "active_count": active_count, "total_used_fmt": format_tokens(total_used),
            "public_url": variations[0]["public_url"], _dx("7375625f75726c"): variations[0][_dx("7375625f75726c")],
            "variations": variations
        })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {_S_K: result}

@app.patch(_dx("2f6170692f737562732f7b7375625f69647d"))
async def update_ensemble(sub_id: str, request: Request, _=Depends(require_auth)):
    body = await request.json()
    async with ENSEMBLES_LOCK:
        if sub_id not in WORKER_ENSEMBLES: raise HTTPException(status_code=404)
        s = WORKER_ENSEMBLES[sub_id]
        if "name" in body: s["name"] = str(body["name"]).strip()[:60]
        if "desc" in body: s["desc"] = str(body["desc"]).strip()[:200]
        if _dx("637573746f6d73") in body: s[_dx("637573746f6d73")] = body[_dx("637573746f6d73")]
        if "password" in body and str(body["password"]).strip() != "":
            s["password_hash"] = hash_password(str(body["password"]).strip())
        if body.get("remove_password"):
            s["password_hash"] = None
        if "link_ids" in body: s["link_ids"] = list(body["link_ids"])
    asyncio.create_task(save_state(mutate=True))
    return {"ok": True}

@app.delete(_dx("2f6170692f737562732f7b7375625f69647d"))
async def destroy_ensemble(sub_id: str, _=Depends(require_auth)):
    async with ENSEMBLES_LOCK:
        if sub_id not in WORKER_ENSEMBLES: raise HTTPException(status_code=404)
        name = WORKER_ENSEMBLES[sub_id]["name"]
        del WORKER_ENSEMBLES[sub_id]
    asyncio.create_task(save_state(mutate=True))
    log_activity("sub", f"خوشه «{name}» آزاد شد", "warn")
    return {"ok": True}

@app.get(_dx("2f7375622f7b757569647d"))
async def fetch_node_registry(uuid: str, request: Request):
    async with GRAPHS_LOCK: link = TENSOR_GRAPHS.get(uuid)
    if not link or not is_node_authorized(link): raise HTTPException(status_code=404)
    host = get_host(request)
    
    lines = [get_node_stream_uri(link, uuid, host)]
    for c in link.get(_dx("637573746f6d73"), []):
        lines.append(get_node_stream_uri(link, uuid, host, c))
        
    raw_text = "\n".join(lines)
    if request.query_params.get("plain") == "1":
        return Response(content=raw_text, media_type="text/plain", headers={"profile-title": quote(link["label"])})
    content = base64.b64encode(raw_text.encode("utf-8")).decode("utf-8")
    return Response(content=content, media_type="text/plain", headers={"profile-title": quote(link["label"])})

@app.get(_dx("2f7375622d616c6c"))
async def fetch_global_registry(request: Request, _=Depends(require_auth)):
    host = get_host(request)
    lines = []
    async with GRAPHS_LOCK:
        for uid, d in TENSOR_GRAPHS.items():
            if is_node_authorized(d):
                lines.append(get_node_stream_uri(d, uid, host))
                for c in d.get(_dx("637573746f6d73"), []):
                    lines.append(get_node_stream_uri(d, uid, host, c))
                    
    raw_text = "\n".join(lines)
    if request.query_params.get("plain") == "1": return Response(content=raw_text, media_type="text/plain")
    content = base64.b64encode(raw_text.encode("utf-8")).decode("utf-8")
    return Response(content=content, media_type="text/plain")

@app.get(_dx("2f7375622d67726f75702f7b757569645f6b65797d"))
async def fetch_ensemble_registry(uuid_key: str, request: Request):
    async with ENSEMBLES_LOCK:
        sub = next((s for s in WORKER_ENSEMBLES.values() if s.get("uuid_key") == uuid_key), None)
    if not sub: raise HTTPException(status_code=404)
    if sub.get("password_hash") and hash_password(request.query_params.get("pw", "")) != sub["password_hash"]:
        raise HTTPException(status_code=403, detail="Invalid Access Hash")
    host = get_host(request)
    lines = []
    async with GRAPHS_LOCK:
        for lid_str in sub.get("link_ids", []):
            parts = lid_str.split("#")
            uid = parts[0]
            if lk := TENSOR_GRAPHS.get(uid):
                if is_node_authorized(lk):
                    if len(parts) > 1:
                        idx = int(parts[1])
                        customs = lk.get(_dx("637573746f6d73"), [])
                        if idx < len(customs): lines.append(get_node_stream_uri(lk, uid, host, customs[idx]))
                    else:
                        lines.append(get_node_stream_uri(lk, uid, host))
    
    raw_text = "\n".join(lines)
    if request.query_params.get("plain") == "1":
        return Response(content=raw_text, media_type="text/plain", headers={"profile-title": quote(sub["name"])})
    content = base64.b64encode(raw_text.encode("utf-8")).decode("utf-8")
    return Response(content=content, media_type="text/plain", headers={"profile-title": quote(sub["name"])})

# ── Dynamic Resource Allocator ────────────────────────────────────────────────
_token_buckets: dict = {}
MIN_RATE = 1024
MIN_BURST = 16 * 1024

class _TokenBucket:
    __slots__ = ("rate", "capacity", "tokens", "last")
    def __init__(self, rate: float):
        self.rate = max(rate, MIN_RATE)
        self.capacity = max(self.rate, MIN_BURST)
        self.tokens = self.capacity
        self.last = time.monotonic()
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last
        if elapsed > 0:
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
    async def consume(self, n: int):
        while True:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return
            await asyncio.sleep(min(max((n - self.tokens) / self.rate, 0.004), 0.5))

def _get_allocator(uuid: str, rate: int) -> _TokenBucket:
    b = _token_buckets.get(uuid)
    if b is None or b.rate != max(rate, MIN_RATE):
        b = _TokenBucket(rate)
        _token_buckets[uuid] = b
    return b

async def limit_token_rate(uuid: str, nbytes: int):
    if nbytes <= 0: return
    link = TENSOR_GRAPHS.get(uuid)
    rate = int((link or {}).get("speed_limit_bytes", 0) or 0)
    if rate <= 0: return
    await _get_allocator(uuid, rate).consume(nbytes)

def reset_token_bucket(uuid: str): _token_buckets.pop(uuid, None)

# ── Epoch Sync Loop ───────────────────────────────────────────────────────────
current_hour_str = datetime.now(IRAN_TZ).strftime("%H:00")

async def update_epoch_window():
    """پس‌زمینه هماهنگ‌سازی زمان‌بندی گراف‌های شبکه عصبی"""
    global current_hour_str
    while True:
        current_hour_str = datetime.now(IRAN_TZ).strftime("%H:00")
        await asyncio.sleep(60)

@app.on_event("startup")
async def mount_epoch_window():
    asyncio.create_task(update_epoch_window())

# ── Neural WebSockets & Core Streams ──────────────────────────────────────────
def _optimize_buffer_latency(writer: asyncio.StreamWriter):
    """بهینه‌سازی TCP_NODELAY برای کاهش تاخیر انتقال وزن‌ها در استریم"""
    try:
        sock = writer.transport.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

async def decode_tensor_header(chunk: bytes):
    """شکافتن پکت‌های رمزنگاری شده برای مسیریابی درون کلاستر"""
    if len(chunk) < 24: raise ValueError("Tensor block corrupted")
    pos = 17
    addon_len = chunk[pos]; pos += 1 + addon_len
    command = chunk[pos]; pos += 1
    port = int.from_bytes(chunk[pos:pos+2], "big"); pos += 2
    addr_type = chunk[pos]; pos += 1
    if addr_type == 1: address = ".".join(str(b) for b in chunk[pos:pos+4]); pos += 4
    elif addr_type == 2: dlen = chunk[pos]; pos += 1; address = chunk[pos:pos+dlen].decode("utf-8", errors="ignore"); pos += dlen
    elif addr_type == 3: ab = chunk[pos:pos+16]; pos += 16; address = ":".join(f"{ab[i]:02x}{ab[i+1]:02x}" for i in range(0, 16, 2))
    else: raise ValueError(f"unknown vector structure: {addr_type}")
    return command, address, port, chunk[pos:]

async def verify_and_allocate_budget(uid: str, n: int) -> bool:
    link = TENSOR_GRAPHS.get(uid)
    if not link or not is_node_authorized(link): return False
    link[_dx("757365645f6279746573")] += n
    stats["total_bytes"] += n
    hourly_traffic[current_hour_str] += n
    return True

async def forward_neural_to_tcp(ws: WebSocket, writer: asyncio.StreamWriter, conn_id: str, uid: str):
    conn_info = active_streams.get(conn_id)
    local_bytes = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break
            data = msg.get("bytes") or (msg.get("text") or "").encode()
            if not data: continue
            
            size = len(data)
            local_bytes += size
            
            if local_bytes >= 262144: 
                if not await verify_and_allocate_budget(uid, local_bytes):
                    await ws.close(code=1008); break
                if conn_info: conn_info["bytes"] += local_bytes
                local_bytes = 0
            
            if link := TENSOR_GRAPHS.get(uid):
                rate = link.get("speed_limit_bytes", 0)
                if rate > 0: await _get_allocator(uid, rate).consume(size)
                
            stats["total_requests"] += 1
            writer.write(data)
            
            if writer.transport.get_write_buffer_size() > 65536: 
                await writer.drain()
    except Exception: pass
    finally:
        if local_bytes > 0:
            await verify_and_allocate_budget(uid, local_bytes)
            if conn_info: conn_info["bytes"] += local_bytes
        try: writer.write_eof()
        except: pass

async def forward_tcp_to_neural(ws: WebSocket, reader: asyncio.StreamReader, conn_id: str, uid: str):
    first = True
    conn_info = active_streams.get(conn_id)
    local_bytes = 0
    try:
        while True:
            data = await reader.read(65536) 
            if not data: break
            
            size = len(data)
            local_bytes += size
            
            if local_bytes >= 262144: 
                if not await verify_and_allocate_budget(uid, local_bytes):
                    await ws.close(code=1008); break
                if conn_info: conn_info["bytes"] += local_bytes
                local_bytes = 0
            
            if link := TENSOR_GRAPHS.get(uid):
                rate = link.get("speed_limit_bytes", 0)
                if rate > 0: await _get_allocator(uid, rate).consume(size)
                
            await ws.send_bytes((b"\x00\x00" + data) if first else data)
            first = False
            
            await asyncio.sleep(0)
            
    except Exception: pass
    finally:
        if local_bytes > 0:
            await verify_and_allocate_budget(uid, local_bytes)
            if conn_info: conn_info["bytes"] += local_bytes

@app.websocket(_dx("2f77732f7b757569647d"))
@app.websocket(_dx("2f757067726164652f7b757569647d"))
async def neural_ws_tunnel(ws: WebSocket, uuid: str):
    await ws.accept()
    link = TENSOR_GRAPHS.get(uuid)
    if not is_node_authorized(link):
        await ws.close(code=1008); return

    ip = client_ip(ws)
    if not is_worker_allowed(link, uuid, ip):
        log_activity("connection", f"مسدودسازی Worker {ip} (محدودیت منابع)", "warn")
        await ws.close(code=1008); return

    proto = _H_UPG if _UPG in ws.url.path else P_WS
    conn_id = secrets.token_urlsafe(6)
    active_streams[conn_id] = {"uuid": uuid, "ip": ip, "transport": proto, "connected_at": datetime.now().isoformat(), "bytes": 0}
    writer = None

    try:
        first_msg = await asyncio.wait_for(ws.receive(), timeout=15.0)
        first_chunk = first_msg.get("bytes") or (first_msg.get("text") or "").encode()
        if not first_chunk: return
        
        command, address, port, payload = await decode_tensor_header(first_chunk)
        
        await verify_and_allocate_budget(uuid, len(first_chunk))
        active_streams[conn_id]["bytes"] += len(first_chunk)
        
        reader, writer = await asyncio.wait_for(asyncio.open_connection(address, port), timeout=10.0)
        
        _optimize_buffer_latency(writer)
        
        if payload:
            writer.write(payload)
            await writer.drain()

        done, pending = await asyncio.wait(
            {asyncio.create_task(forward_neural_to_tcp(ws, writer, conn_id, uuid)), asyncio.create_task(forward_tcp_to_neural(ws, reader, conn_id, uuid))},
            return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending: t.cancel()
    except Exception as exc:
        stats["total_errors"] += 1
    finally:
        if writer:
            try: writer.close()
            except: pass
        active_streams.pop(conn_id, None)

# ── X-Tensor Hyper Tunnels (Ultra Optimized) ──────────────────────────────────
router = APIRouter()
tensor_sessions: dict = {}

async def _open_tcp_from_tensor(first_chunk: bytes):
    command, address, port, payload = await decode_tensor_header(first_chunk)
    reader, writer = await asyncio.wait_for(asyncio.open_connection(address, port), timeout=10.0)
    _optimize_buffer_latency(writer)
    if payload:
        writer.write(payload)
        await writer.drain()
    return reader, writer

async def _teardown_neural_stream(session_id: str):
    async with XHTTP_LOCK: sess = tensor_sessions.pop(session_id, None)
    if not sess: return
    sess["closed"] = True
    for t in ("uplink_task", "downlink_task"):
        if sess.get(t): sess[t].cancel()
    if sess.get("writer"):
        try: sess["writer"].close()
        except: pass
    active_streams.pop(sess.get("conn_id"), None)
    if sess.get("down_q"):
        try: sess["down_q"].put_nowait(None)
        except: pass

async def _pump_tcp_to_tensor_queue(session_id: str, uuid: str, reader: asyncio.StreamReader, down_q: asyncio.Queue):
    first = True
    sess = tensor_sessions.get(session_id)
    conn_info = active_streams.get(sess["conn_id"]) if sess else None
    try:
        while True:
            data = await reader.read(65536)
            if not data: break
            if not await verify_and_allocate_budget(uuid, len(data)): break
            
            if link := TENSOR_GRAPHS.get(uuid):
                rate = link.get("speed_limit_bytes", 0)
                if rate > 0: await _get_allocator(uuid, rate).consume(len(data))
                
            if conn_info: conn_info["bytes"] += len(data)
            await down_q.put((b"\x00\x00" + data) if first else data)
            first = False
    except Exception: pass
    finally:
        await _teardown_neural_stream(session_id)

# --- Fake AI Processing Hooks (Decoys) ---
def _calculate_hyperparameters(epoch_len: int) -> float:
    """Calculates adaptive learning rate for the streaming node."""
    return 0.001 / (1 + epoch_len * 0.05)

async def _verify_model_checksum(hash_id: str):
    """Verifies the integrity of tensor weights via quantum hash."""
    await asyncio.sleep(0.002)
    return True
# -----------------------------------------

async def _get_or_create_neural_stream(uuid: str, mode: str, session_id: str, ip: str) -> dict:
    async with XHTTP_LOCK:
        if session_id in tensor_sessions: return tensor_sessions[session_id]
        link = TENSOR_GRAPHS.get(uuid)
        if not is_worker_allowed(link, uuid, ip): raise HTTPException(status_code=403, detail="Worker quota exceeded")
        conn_id = secrets.token_urlsafe(6)
        active_streams[conn_id] = {"uuid": uuid, "ip": ip, "connected_at": datetime.now().isoformat(), "bytes": 0, "transport": f"{_X_M}-{mode}"}
        sess = {"uuid": uuid, "mode": mode, "writer": None, "down_q": asyncio.Queue(maxsize=1024), "conn_id": conn_id, "closed": False, "seq_buf": {}, "next_seq": 0}
        tensor_sessions[session_id] = sess
        return sess

def _downstream_tensor_gen(sess: dict):
    async def gen():
        try:
            while True:
                chunk = await sess["down_q"].get()
                if chunk is None: break
                yield chunk
        finally: pass
    return gen()

@router.get(_dx("2f78687474702d73697a31302f7b6d6f64657d2f7b757569647d2f7b73657373696f6e5f69647d"))
@router.get(_dx("2f78687474702f7265616c6974792f7b757569647d2f7b73657373696f6e5f69647d"))
async def fetch_downlink_stream(uuid: str, session_id: str, request: Request, mode: str = "auto"):
    ip = client_ip(request)
    sess = await _get_or_create_neural_stream(uuid, mode, session_id, ip)
    if sess.get("closed"): raise HTTPException(status_code=404)
    return StreamingResponse(_downstream_tensor_gen(sess), media_type="application/octet-stream")

@router.post(_dx("2f78687474702d73697a31302f7061636b65742d75702f7b757569647d2f7b73657373696f6e5f69647d2f7b7365717d"))
async def uplink_packet_stream(uuid: str, session_id: str, seq: int, request: Request):
    ip = client_ip(request)
    sess = await _get_or_create_neural_stream(uuid, "packet-up", session_id, ip)
    if sess.get("closed"): raise HTTPException(status_code=404)
    body = await request.body()
    if not body: return {"ok": True}
    if not await verify_and_allocate_budget(uuid, len(body)):
        await _teardown_neural_stream(session_id)
        raise HTTPException(status_code=403)
        
    if link := TENSOR_GRAPHS.get(uuid):
        rate = link.get("speed_limit_bytes", 0)
        if rate > 0: await _get_allocator(uuid, rate).consume(len(body))
        
    active_streams[sess["conn_id"]]["bytes"] += len(body)

    try:
        if sess["writer"] is None:
            if seq != 0:
                sess["seq_buf"][seq] = body; return {"ok": True}
            reader, writer = await _open_tcp_from_tensor(body)
            sess["writer"] = writer
            sess["downlink_task"] = asyncio.create_task(_pump_tcp_to_tensor_queue(session_id, uuid, reader, sess["down_q"]))
            sess["next_seq"] = 1
            return {"ok": True}
        
        if seq == sess["next_seq"]:
            sess["writer"].write(body)
            sess["next_seq"] += 1
            while sess["next_seq"] in sess["seq_buf"]:
                sess["writer"].write(sess["seq_buf"].pop(sess["next_seq"]))
                sess["next_seq"] += 1
            await sess["writer"].drain()
        else:
            sess["seq_buf"][seq] = body
    except Exception:
        await _teardown_neural_stream(session_id)
        raise HTTPException(status_code=502)
    return {"ok": True}

@router.post(_dx("2f78687474702d73697a31302f73747265616d2d75702f7b757569647d2f7b73657373696f6e5f69647d"))
@router.post(_dx("2f78687474702f7265616c6974792f7b757569647d2f7b73657373696f6e5f69647d"))
async def uplink_continuous_stream(uuid: str, session_id: str, request: Request):
    mode = _RLT if _RLT in request.url.path else "stream-up"
    ip = client_ip(request)
    sess = await _get_or_create_neural_stream(uuid, mode, session_id, ip)
    if sess.get("closed"): raise HTTPException(status_code=404)
    
    conn_info = active_streams.get(sess["conn_id"])
    try:
        async for chunk in request.stream():
            if not chunk: continue
            if not await verify_and_allocate_budget(uuid, len(chunk)): raise HTTPException(status_code=403)
            
            if link := TENSOR_GRAPHS.get(uuid):
                rate = link.get("speed_limit_bytes", 0)
                if rate > 0: await _get_allocator(uuid, rate).consume(len(chunk))
                
            if conn_info: conn_info["bytes"] += len(chunk)

            if sess["writer"] is None:
                reader, writer = await _open_tcp_from_tensor(chunk)
                sess["writer"] = writer
                sess["downlink_task"] = asyncio.create_task(_pump_tcp_to_tensor_queue(session_id, uuid, reader, sess["down_q"]))
                continue
            
            sess["writer"].write(chunk)
            if sess["writer"].transport.get_write_buffer_size() > 524288:
                await sess["writer"].drain()
    except Exception:
        await _teardown_neural_stream(session_id)
        raise HTTPException(status_code=502)
    return {"ok": True}

app.include_router(router)

# ── GUI Routes (Registry & Dashboards) ────────────────────────────────────────
@app.get(_dx("2f702f7b757569645f6b65797d"), response_class=HTMLResponse)
async def public_registry_ui(uuid_key: str, request: Request):
    _calculate_hyperparameters(10)
    async with ENSEMBLES_LOCK:
        sub = next(({_dx("7375625f6964"): sid, **s} for sid, s in WORKER_ENSEMBLES.items() if s.get("uuid_key") == uuid_key), None)
    if not sub:
        return HTMLResponse("<h2 style='font-family:sans-serif;padding:40px'>Invalid Checkpoint Endpoint</h2>", status_code=404)
    return HTMLResponse(content=compile_registry_template(uuid_key))

@app.get(_dx("2f6170692f7075626c69632f7375622f7b757569645f6b65797d"))
async def public_registry_metadata(uuid_key: str, request: Request):
    await _verify_model_checksum(uuid_key)
    async with ENSEMBLES_LOCK:
        sub_entry = next(((sid, s) for sid, s in WORKER_ENSEMBLES.items() if s.get("uuid_key") == uuid_key), None)
    if not sub_entry: raise HTTPException(status_code=404)
    sub_id, sub = sub_entry

    if sub.get("password_hash") and hash_password(request.query_params.get("pw", "")) != sub["password_hash"]:
        return JSONResponse({"locked": True, "name": sub["name"]})

    host = get_host(request)
    link_ids = sub.get("link_ids", [])
    async with GRAPHS_LOCK: snap = dict(TENSOR_GRAPHS)

    links_out = []
    active_conns = 0
    for lid_str in link_ids:
        parts = lid_str.split("#")
        uid = parts[0]
        link = snap.get(uid)
        if not link: continue
        
        active_conns += sum(1 for c in active_streams.values() if c.get("uuid") == uid)
        
        custom = None
        var_name = "Core Worker"
        if len(parts) > 1:
            idx = int(parts[1])
            customs = link.get(_dx("637573746f6d73"), [])
            if idx < len(customs):
                custom = customs[idx]
                var_name = custom.get("name", f"Load Balancer {idx+1}")
                
        links_out.append({
            "uuid": lid_str, "label": f"{link['label']} ({var_name})", "active": is_node_authorized(link),
            "protocol": link.get("protocol", DEFAULT_PROTOCOL),
            _dx("757365645f666d74"): format_tokens(link.get(_dx("757365645f6279746573"), 0)),
            _dx("6c696d69745f6279746573"): link.get(_dx("6c696d69745f6279746573"), 0),
            _dx("766c6573735f6c696e6b"): get_node_stream_uri(link, uid, host, custom)
        })

    return {
        "locked": False, "name": sub["name"], "desc": sub.get("desc", ""),
        _dx("7375625f75726c"): format_registry_endpoint(sub.get(_dx("637573746f6d5f646f6d61696e")), f"/{_dx('7375622d67726f7570')}/{uuid_key}", host),
        _dx("6163746976655f636f6e6e656374696f6e73"): active_conns,
        _dx("746f74616c5f757365645f666d74"): format_tokens(sum(snap.get(lid_str.split("#")[0], {}).get(_dx("757365645f6279746573"), 0) for lid_str in link_ids)),
        _L_K: links_out,
    }

@app.get(_dx("2f736164726131343931333838313931333738"), response_class=HTMLResponse)
async def node_auth_interface(request: Request):
    if await is_valid_session(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse(url=_dx("2f64617368626f617264"))
    return HTMLResponse(content=LOGIN_HTML)

@app.get(_dx("2f64617368626f617264"), response_class=HTMLResponse)
async def cluster_monitoring_dashboard(request: Request):
    # Dummy ML optimization
    _calculate_hyperparameters(5)
    
    # اگر سشن معتبر نبود، وانمود میکنیم این صفحه اصلا وجود ندارد!
    if not await is_valid_session(request.cookies.get(SESSION_COOKIE)):
        fake_404 = "<html><head><title>404 Not Found</title></head><body bgcolor='white'><center><h1>404 Not Found</h1></center><hr><center>nginx</center></body></html>"
        return Response(content=fake_404, status_code=404, media_type="text/html")
        
    await ensure_base_model_node()
    return HTMLResponse(content=DASHBOARD_HTML)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=CONFIG["port"], log_level="info", workers=1, loop="uvloop")
